"""Custom Metal LSTM cell kernels via mx.fast.metal_kernel.

Ported from the in-progress MLX research branch (Godin, PR ml-explore/mlx#3089
plus tuning commit `0b6632f1e`). Adapted to user-land `mx.fast.metal_kernel`
so it ships without an MLX core rebuild.

Three flavours of cell kernel:
- `lstm_cell_kernel`              : single LSTM, fast math (default).
- `lstm_cell_precise_kernel`      : single LSTM, `metal::precise::{tanh,exp}`
                                    for training-quality accuracy.
- `grouped_lstm_cell_kernel`      : G parallel LSTMs in one launch.

Threadgroup sizing matches the C++ `pick_threads_per_group` heuristic from
the research branch (key tuning beyond what's in the open PR).

VJP (training) kernel exists in the research branch but is not yet wired here
— `mx.fast.metal_kernel` doesn't expose VJPs directly; the next iteration
would register via `mx.custom_function`.
"""
from __future__ import annotations

import os

try:
    import mlx.core as mx
    import mlx.nn as mlxnn
except ImportError as e:
    raise ImportError("pip install mlx") from e


# ============================================================================
# Tuning heuristic — port of pick_threads_per_group from fast_lstm_cell.cpp
# ============================================================================

# Apple Silicon GPUs have maxTotalThreadsPerThreadgroup = 1024
_APPLE_MAX_TG = 1024


def _pick_threads_per_group(hidden_size: int, batch_size: int, total_threads: int) -> int:
    if hidden_size >= 512 or batch_size >= 512:
        target = 1024
    elif hidden_size >= 256 or batch_size >= 128:
        target = 512
    elif hidden_size >= 64 or batch_size >= 32:
        target = 256
    else:
        target = 128
    tg = min(target, _APPLE_MAX_TG)
    # Round down to SIMD width (32)
    if tg >= 32:
        tg = (tg // 32) * 32
    return max(1, min(tg, total_threads))


def _env_use_precise() -> bool:
    """Mirror MLX_FAST_LSTM_PRECISE_MATH env var from research branch."""
    v = os.environ.get("MLX_FAST_LSTM_PRECISE_MATH", "")
    return v[:1] in ("1", "t", "T")


# ============================================================================
# Header — shared by all kernels
# ============================================================================

_HEADER = """
#include <metal_math>
#include <metal_stdlib>
using namespace metal;

inline float fast_sigmoid_mlxa(float x) {
    float y = 1.0f / (1.0f + metal::fast::exp(-metal::abs(x)));
    return (x < 0.0f) ? 1.0f - y : y;
}

inline float4 fast_sigmoid4_mlxa(float4 x) {
    return float4(
        fast_sigmoid_mlxa(x.x), fast_sigmoid_mlxa(x.y),
        fast_sigmoid_mlxa(x.z), fast_sigmoid_mlxa(x.w));
}

inline float stable_sigmoid_mlxa(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + metal::precise::exp(-x));
    }
    float ex = metal::precise::exp(x);
    return ex / (1.0f + ex);
}

inline float4 stable_sigmoid4_mlxa(float4 x) {
    return float4(
        stable_sigmoid_mlxa(x.x), stable_sigmoid_mlxa(x.y),
        stable_sigmoid_mlxa(x.z), stable_sigmoid_mlxa(x.w));
}
"""


def _make_single_kernel_source(precise: bool) -> str:
    """Emit MSL source with either fast or precise activations baked in."""
    sigmoid = "stable_sigmoid_mlxa" if precise else "fast_sigmoid_mlxa"
    sigmoid4 = "stable_sigmoid4_mlxa" if precise else "fast_sigmoid4_mlxa"
    tanh = "metal::precise::tanh" if precise else "metal::fast::tanh"
    return f"""
    uint hidden_size = cell_prev_shape[1];
    uint batch_size = cell_prev_shape[0];
    uint stride_4h = 4u * hidden_size;
    uint h_quads = (hidden_size + 3u) / 4u;
    uint total_quads = batch_size * h_quads;

    uint idx = thread_position_in_grid.x;
    if (idx >= total_quads) return;

    uint batch_idx = idx / h_quads;
    uint h_base = (idx % h_quads) * 4u;
    uint base = batch_idx * stride_4h + h_base;
    uint prev_base = batch_idx * hidden_size + h_base;

    if (h_base + 4u <= hidden_size) {{
        float4 i4 = {sigmoid4}(
            *reinterpret_cast<const device float4*>(input_proj + base) +
            *reinterpret_cast<const device float4*>(hidden_proj + base));
        float4 f4 = {sigmoid4}(
            *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + hidden_size));
        float4 g4 = {tanh}(
            *reinterpret_cast<const device float4*>(input_proj + base + 2u * hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + 2u * hidden_size));
        float4 o4 = {sigmoid4}(
            *reinterpret_cast<const device float4*>(input_proj + base + 3u * hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + 3u * hidden_size));
        float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);
        float4 c_new4 = f4 * c_prev4 + i4 * g4;
        *reinterpret_cast<device float4*>(output_cell + prev_base) = c_new4;
        *reinterpret_cast<device float4*>(output_hidden + prev_base) =
            o4 * {tanh}(c_new4);
        return;
    }}

    for (uint k = 0u; k < 4u && (h_base + k) < hidden_size; k++) {{
        uint b = base + k;
        uint pb = prev_base + k;
        float i_g = {sigmoid}(input_proj[b] + hidden_proj[b]);
        float f_g = {sigmoid}(input_proj[b + hidden_size] + hidden_proj[b + hidden_size]);
        float g_g = {tanh}(input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size]);
        float o_g = {sigmoid}(input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size]);
        float c_prev = cell_prev[pb];
        float c_new = f_g * c_prev + i_g * g_g;
        output_cell[pb] = c_new;
        output_hidden[pb] = o_g * {tanh}(c_new);
    }}
"""


_LSTM_CELL_KERNEL_FAST = mx.fast.metal_kernel(
    name="mlxa_lstm_cell_fast",
    input_names=["input_proj", "hidden_proj", "cell_prev"],
    output_names=["output_cell", "output_hidden"],
    source=_make_single_kernel_source(precise=False),
    header=_HEADER,
    ensure_row_contiguous=True,
)

_LSTM_CELL_KERNEL_PRECISE = mx.fast.metal_kernel(
    name="mlxa_lstm_cell_precise",
    input_names=["input_proj", "hidden_proj", "cell_prev"],
    output_names=["output_cell", "output_hidden"],
    source=_make_single_kernel_source(precise=True),
    header=_HEADER,
    ensure_row_contiguous=True,
)


def _lstm_cell_step(input_proj, hidden_proj, cell_prev, precise: bool = False):
    B, fourH = input_proj.shape
    H = fourH // 4
    h_quads = (H + 3) // 4
    total = B * h_quads
    tg = _pick_threads_per_group(H, B, total)
    kernel = _LSTM_CELL_KERNEL_PRECISE if precise else _LSTM_CELL_KERNEL_FAST
    return kernel(
        inputs=[input_proj, hidden_proj, cell_prev],
        output_shapes=[(B, H), (B, H)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(total, 1, 1),
        threadgroup=(tg, 1, 1),
    )


# ============================================================================
# VJP kernel — backward pass for the single LSTM cell
# ============================================================================
# Math (chain rule through h_new = o * tanh(c_new) and c_new = f * c_prev + i * g):
#   dc = cot_c + cot_h * o * (1 - tanh(c_new)^2)
#   do = cot_h * tanh(c_new)
#   di = dc * g     ;   df = dc * c_prev   ;   dg = dc * i
#   d_i_gate = di * i * (1 - i)              (sigmoid backward)
#   d_f_gate = df * f * (1 - f)
#   d_g_gate = dg * (1 - g^2)                (tanh backward)
#   d_o_gate = do * o * (1 - o)
#   d_input_proj = d_hidden_proj = [d_i_gate, d_f_gate, d_g_gate, d_o_gate]
#   d_cell_prev  = dc * f

def _make_vjp_kernel_source(precise: bool) -> str:
    sigmoid4 = "stable_sigmoid4_mlxa" if precise else "fast_sigmoid4_mlxa"
    sigmoid = "stable_sigmoid_mlxa" if precise else "fast_sigmoid_mlxa"
    tanh = "metal::precise::tanh" if precise else "metal::fast::tanh"
    return f"""
    uint hidden_size = cell_prev_shape[1];
    uint batch_size = cell_prev_shape[0];
    uint stride_4h = 4u * hidden_size;
    uint h_quads = (hidden_size + 3u) / 4u;
    uint total_quads = batch_size * h_quads;
    uint idx = thread_position_in_grid.x;
    if (idx >= total_quads) return;
    uint batch_idx = idx / h_quads;
    uint h_base = (idx % h_quads) * 4u;
    uint base = batch_idx * stride_4h + h_base;
    uint prev_base = batch_idx * hidden_size + h_base;

    if (h_base + 4u <= hidden_size) {{
        float4 gi4 = *reinterpret_cast<const device float4*>(input_proj + base) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base);
        float4 gf4 = *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base + hidden_size);
        float4 gg4 = *reinterpret_cast<const device float4*>(input_proj + base + 2u * hidden_size) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base + 2u * hidden_size);
        float4 go4 = *reinterpret_cast<const device float4*>(input_proj + base + 3u * hidden_size) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base + 3u * hidden_size);

        float4 i4 = {sigmoid4}(gi4);
        float4 f4 = {sigmoid4}(gf4);
        float4 g4 = {tanh}(gg4);
        float4 o4 = {sigmoid4}(go4);

        float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);
        float4 c_new4 = f4 * c_prev4 + i4 * g4;
        float4 tanh_c4 = {tanh}(c_new4);

        float4 cot_c4 = *reinterpret_cast<const device float4*>(cot_cell + prev_base);
        float4 cot_h4 = *reinterpret_cast<const device float4*>(cot_hidden + prev_base);

        float4 dc4 = cot_c4 + cot_h4 * o4 * (1.0f - tanh_c4 * tanh_c4);
        float4 do4 = cot_h4 * tanh_c4;
        float4 di4 = dc4 * g4;
        float4 df4 = dc4 * c_prev4;
        float4 dg4 = dc4 * i4;

        float4 d_i_gate4 = di4 * i4 * (1.0f - i4);
        float4 d_f_gate4 = df4 * f4 * (1.0f - f4);
        float4 d_g_gate4 = dg4 * (1.0f - g4 * g4);
        float4 d_o_gate4 = do4 * o4 * (1.0f - o4);

        *reinterpret_cast<device float4*>(d_input_proj + base) = d_i_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + hidden_size) = d_f_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + 2u * hidden_size) = d_g_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + 3u * hidden_size) = d_o_gate4;

        *reinterpret_cast<device float4*>(d_hidden_proj + base) = d_i_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + hidden_size) = d_f_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + 2u * hidden_size) = d_g_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + 3u * hidden_size) = d_o_gate4;

        *reinterpret_cast<device float4*>(d_cell_prev + prev_base) = dc4 * f4;
        return;
    }}

    for (uint k = 0u; k < 4u && (h_base + k) < hidden_size; k++) {{
        uint b = base + k;
        uint pb = prev_base + k;
        float gi = input_proj[b] + hidden_proj[b];
        float gf = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
        float gg = input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size];
        float go = input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size];
        float i_g = {sigmoid}(gi);
        float f_g = {sigmoid}(gf);
        float g_g = {tanh}(gg);
        float o_g = {sigmoid}(go);
        float c_prev_v = cell_prev[pb];
        float c_new = f_g * c_prev_v + i_g * g_g;
        float tanh_c = {tanh}(c_new);
        float cot_c_v = cot_cell[pb];
        float cot_h_v = cot_hidden[pb];
        float dc = cot_c_v + cot_h_v * o_g * (1.0f - tanh_c * tanh_c);
        float do_v = cot_h_v * tanh_c;
        float di_v = dc * g_g;
        float df_v = dc * c_prev_v;
        float dg_v = dc * i_g;
        float d_i_gate = di_v * i_g * (1.0f - i_g);
        float d_f_gate = df_v * f_g * (1.0f - f_g);
        float d_g_gate = dg_v * (1.0f - g_g * g_g);
        float d_o_gate = do_v * o_g * (1.0f - o_g);
        d_input_proj[b] = d_i_gate;
        d_input_proj[b + hidden_size] = d_f_gate;
        d_input_proj[b + 2u * hidden_size] = d_g_gate;
        d_input_proj[b + 3u * hidden_size] = d_o_gate;
        d_hidden_proj[b] = d_i_gate;
        d_hidden_proj[b + hidden_size] = d_f_gate;
        d_hidden_proj[b + 2u * hidden_size] = d_g_gate;
        d_hidden_proj[b + 3u * hidden_size] = d_o_gate;
        d_cell_prev[pb] = dc * f_g;
    }}
"""


_LSTM_CELL_VJP_KERNEL_FAST = mx.fast.metal_kernel(
    name="mlxa_lstm_cell_vjp_fast",
    input_names=["input_proj", "hidden_proj", "cell_prev", "cot_cell", "cot_hidden"],
    output_names=["d_input_proj", "d_hidden_proj", "d_cell_prev"],
    source=_make_vjp_kernel_source(precise=False),
    header=_HEADER,
    ensure_row_contiguous=True,
)

_LSTM_CELL_VJP_KERNEL_PRECISE = mx.fast.metal_kernel(
    name="mlxa_lstm_cell_vjp_precise",
    input_names=["input_proj", "hidden_proj", "cell_prev", "cot_cell", "cot_hidden"],
    output_names=["d_input_proj", "d_hidden_proj", "d_cell_prev"],
    source=_make_vjp_kernel_source(precise=True),
    header=_HEADER,
    ensure_row_contiguous=True,
)


def _lstm_cell_vjp_step(input_proj, hidden_proj, cell_prev, cot_cell, cot_hidden,
                        precise: bool = False):
    B, fourH = input_proj.shape
    H = fourH // 4
    h_quads = (H + 3) // 4
    total = B * h_quads
    tg = _pick_threads_per_group(H, B, total)
    kernel = _LSTM_CELL_VJP_KERNEL_PRECISE if precise else _LSTM_CELL_VJP_KERNEL_FAST
    return kernel(
        inputs=[input_proj, hidden_proj, cell_prev, cot_cell, cot_hidden],
        output_shapes=[(B, 4 * H), (B, 4 * H), (B, H)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(total, 1, 1),
        threadgroup=(tg, 1, 1),
    )


# ============================================================================
# Differentiable cell — forward + VJP wired through mx.custom_function
# ============================================================================

@mx.custom_function
def metal_lstm_cell(input_proj, hidden_proj, cell_prev):
    """LSTM cell with autograd-aware Metal kernels (forward + VJP).

    Use this inside a Python time loop instead of _lstm_cell_step when you need
    `mx.grad(...)` to flow through. For inference-only paths use _lstm_cell_step
    directly (one fewer Python wrapper)."""
    return _lstm_cell_step(input_proj, hidden_proj, cell_prev, precise=False)


@metal_lstm_cell.vjp
def _metal_lstm_cell_vjp(primals, cotangents, outputs):
    input_proj, hidden_proj, cell_prev = primals
    cot_cell, cot_hidden = cotangents
    return _lstm_cell_vjp_step(input_proj, hidden_proj, cell_prev, cot_cell, cot_hidden,
                               precise=False)


# ============================================================================
# Residual-saving variants — forward saves (gates, tanh_c) so VJP avoids
# recomputing the 5 expensive activations per quad.
# ============================================================================

_SOURCE_FORWARD_WITH_RESIDUALS = """
    uint hidden_size = cell_prev_shape[1];
    uint batch_size = cell_prev_shape[0];
    uint stride_4h = 4u * hidden_size;
    uint h_quads = (hidden_size + 3u) / 4u;
    uint total_quads = batch_size * h_quads;
    uint idx = thread_position_in_grid.x;
    if (idx >= total_quads) return;
    uint batch_idx = idx / h_quads;
    uint h_base = (idx % h_quads) * 4u;
    uint base = batch_idx * stride_4h + h_base;
    uint prev_base = batch_idx * hidden_size + h_base;

    if (h_base + 4u <= hidden_size) {
        float4 i4 = fast_sigmoid4_mlxa(
            *reinterpret_cast<const device float4*>(input_proj + base) +
            *reinterpret_cast<const device float4*>(hidden_proj + base));
        float4 f4 = fast_sigmoid4_mlxa(
            *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + hidden_size));
        float4 g4 = metal::fast::tanh(
            *reinterpret_cast<const device float4*>(input_proj + base + 2u * hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + 2u * hidden_size));
        float4 o4 = fast_sigmoid4_mlxa(
            *reinterpret_cast<const device float4*>(input_proj + base + 3u * hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + 3u * hidden_size));
        float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);
        float4 c_new4 = f4 * c_prev4 + i4 * g4;
        float4 tanh_c4 = metal::fast::tanh(c_new4);
        *reinterpret_cast<device float4*>(output_cell + prev_base) = c_new4;
        *reinterpret_cast<device float4*>(output_hidden + prev_base) = o4 * tanh_c4;
        // Save residuals: gates packed as (B, 4H) in same layout as input_proj
        *reinterpret_cast<device float4*>(residuals_gates + base) = i4;
        *reinterpret_cast<device float4*>(residuals_gates + base + hidden_size) = f4;
        *reinterpret_cast<device float4*>(residuals_gates + base + 2u * hidden_size) = g4;
        *reinterpret_cast<device float4*>(residuals_gates + base + 3u * hidden_size) = o4;
        *reinterpret_cast<device float4*>(residuals_tanh_c + prev_base) = tanh_c4;
        return;
    }

    for (uint k = 0u; k < 4u && (h_base + k) < hidden_size; k++) {
        uint b = base + k;
        uint pb = prev_base + k;
        float i_g = fast_sigmoid_mlxa(input_proj[b] + hidden_proj[b]);
        float f_g = fast_sigmoid_mlxa(input_proj[b + hidden_size] + hidden_proj[b + hidden_size]);
        float g_g = metal::fast::tanh(input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size]);
        float o_g = fast_sigmoid_mlxa(input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size]);
        float c_prev = cell_prev[pb];
        float c_new = f_g * c_prev + i_g * g_g;
        float tanh_c = metal::fast::tanh(c_new);
        output_cell[pb] = c_new;
        output_hidden[pb] = o_g * tanh_c;
        residuals_gates[b] = i_g;
        residuals_gates[b + hidden_size] = f_g;
        residuals_gates[b + 2u * hidden_size] = g_g;
        residuals_gates[b + 3u * hidden_size] = o_g;
        residuals_tanh_c[pb] = tanh_c;
    }
"""

_LSTM_CELL_FWD_RES_KERNEL = mx.fast.metal_kernel(
    name="mlxa_lstm_cell_fwd_residuals",
    input_names=["input_proj", "hidden_proj", "cell_prev"],
    output_names=["output_cell", "output_hidden", "residuals_gates", "residuals_tanh_c"],
    source=_SOURCE_FORWARD_WITH_RESIDUALS,
    header=_HEADER,
    ensure_row_contiguous=True,
)


def _lstm_cell_step_fwd_residuals(input_proj, hidden_proj, cell_prev):
    B, fourH = input_proj.shape
    H = fourH // 4
    h_quads = (H + 3) // 4
    total = B * h_quads
    tg = _pick_threads_per_group(H, B, total)
    return _LSTM_CELL_FWD_RES_KERNEL(
        inputs=[input_proj, hidden_proj, cell_prev],
        output_shapes=[(B, H), (B, H), (B, 4 * H), (B, H)],
        output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
        grid=(total, 1, 1),
        threadgroup=(tg, 1, 1),
    )


_SOURCE_VJP_FROM_RESIDUALS = """
    uint hidden_size = cell_prev_shape[1];
    uint batch_size = cell_prev_shape[0];
    uint stride_4h = 4u * hidden_size;
    uint h_quads = (hidden_size + 3u) / 4u;
    uint total_quads = batch_size * h_quads;
    uint idx = thread_position_in_grid.x;
    if (idx >= total_quads) return;
    uint batch_idx = idx / h_quads;
    uint h_base = (idx % h_quads) * 4u;
    uint base = batch_idx * stride_4h + h_base;
    uint prev_base = batch_idx * hidden_size + h_base;

    if (h_base + 4u <= hidden_size) {
        // Read saved residuals (cheap memory reads instead of recompute)
        float4 i4 = *reinterpret_cast<const device float4*>(gates + base);
        float4 f4 = *reinterpret_cast<const device float4*>(gates + base + hidden_size);
        float4 g4 = *reinterpret_cast<const device float4*>(gates + base + 2u * hidden_size);
        float4 o4 = *reinterpret_cast<const device float4*>(gates + base + 3u * hidden_size);
        float4 tanh_c4 = *reinterpret_cast<const device float4*>(tanh_c + prev_base);
        float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);

        float4 cot_c4 = *reinterpret_cast<const device float4*>(cot_cell + prev_base);
        float4 cot_h4 = *reinterpret_cast<const device float4*>(cot_hidden + prev_base);

        float4 dc4 = cot_c4 + cot_h4 * o4 * (1.0f - tanh_c4 * tanh_c4);
        float4 do4 = cot_h4 * tanh_c4;
        float4 di4 = dc4 * g4;
        float4 df4 = dc4 * c_prev4;
        float4 dg4 = dc4 * i4;

        float4 d_i_gate4 = di4 * i4 * (1.0f - i4);
        float4 d_f_gate4 = df4 * f4 * (1.0f - f4);
        float4 d_g_gate4 = dg4 * (1.0f - g4 * g4);
        float4 d_o_gate4 = do4 * o4 * (1.0f - o4);

        *reinterpret_cast<device float4*>(d_input_proj + base) = d_i_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + hidden_size) = d_f_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + 2u * hidden_size) = d_g_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + 3u * hidden_size) = d_o_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base) = d_i_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + hidden_size) = d_f_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + 2u * hidden_size) = d_g_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + 3u * hidden_size) = d_o_gate4;
        *reinterpret_cast<device float4*>(d_cell_prev + prev_base) = dc4 * f4;
        return;
    }

    for (uint k = 0u; k < 4u && (h_base + k) < hidden_size; k++) {
        uint b = base + k;
        uint pb = prev_base + k;
        float i_g = gates[b];
        float f_g = gates[b + hidden_size];
        float g_g = gates[b + 2u * hidden_size];
        float o_g = gates[b + 3u * hidden_size];
        float tanh_c_v = tanh_c[pb];
        float c_prev_v = cell_prev[pb];
        float cot_c_v = cot_cell[pb];
        float cot_h_v = cot_hidden[pb];
        float dc = cot_c_v + cot_h_v * o_g * (1.0f - tanh_c_v * tanh_c_v);
        float do_v = cot_h_v * tanh_c_v;
        float di_v = dc * g_g;
        float df_v = dc * c_prev_v;
        float dg_v = dc * i_g;
        float d_i_gate = di_v * i_g * (1.0f - i_g);
        float d_f_gate = df_v * f_g * (1.0f - f_g);
        float d_g_gate = dg_v * (1.0f - g_g * g_g);
        float d_o_gate = do_v * o_g * (1.0f - o_g);
        d_input_proj[b] = d_i_gate;
        d_input_proj[b + hidden_size] = d_f_gate;
        d_input_proj[b + 2u * hidden_size] = d_g_gate;
        d_input_proj[b + 3u * hidden_size] = d_o_gate;
        d_hidden_proj[b] = d_i_gate;
        d_hidden_proj[b + hidden_size] = d_f_gate;
        d_hidden_proj[b + 2u * hidden_size] = d_g_gate;
        d_hidden_proj[b + 3u * hidden_size] = d_o_gate;
        d_cell_prev[pb] = dc * f_g;
    }
"""

_LSTM_CELL_VJP_RES_KERNEL = mx.fast.metal_kernel(
    name="mlxa_lstm_cell_vjp_residuals",
    input_names=["cell_prev", "gates", "tanh_c", "cot_cell", "cot_hidden"],
    output_names=["d_input_proj", "d_hidden_proj", "d_cell_prev"],
    source=_SOURCE_VJP_FROM_RESIDUALS,
    header=_HEADER,
    ensure_row_contiguous=True,
)


def _lstm_cell_vjp_from_residuals(cell_prev, gates, tanh_c, cot_cell, cot_hidden):
    B, fourH = gates.shape
    H = fourH // 4
    h_quads = (H + 3) // 4
    total = B * h_quads
    tg = _pick_threads_per_group(H, B, total)
    return _LSTM_CELL_VJP_RES_KERNEL(
        inputs=[cell_prev, gates, tanh_c, cot_cell, cot_hidden],
        output_shapes=[(B, 4 * H), (B, 4 * H), (B, H)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(total, 1, 1),
        threadgroup=(tg, 1, 1),
    )


@mx.custom_function
def metal_lstm_cell_v2(input_proj, hidden_proj, cell_prev):
    """Residual-saving variant — forward writes gates+tanh_c so VJP avoids
    recomputing them. Returns 4-tuple (cell_new, hidden_new, gates, tanh_c);
    the residual outputs are consumed only by the VJP.

    NOTE on Apple Silicon: this is NOT measurably faster than the recompute
    variant — `metal::fast::tanh` and `fast_sigmoid` are so cheap that the
    extra memory traffic for the residuals matches their compute cost. We
    ship this for completeness (may help on hardware where transcendentals
    are more expensive, e.g., CUDA SMs without dedicated SFU pipelines)."""
    return _lstm_cell_step_fwd_residuals(input_proj, hidden_proj, cell_prev)


@metal_lstm_cell_v2.vjp
def _metal_lstm_cell_v2_vjp(primals, cotangents, outputs):
    input_proj, hidden_proj, cell_prev = primals
    cot_cell, cot_hidden, _, _ = cotangents  # residual cotangents ignored
    cell_new, hidden_new, gates_res, tanh_c_res = outputs
    d_inp, d_hid, d_prev = _lstm_cell_vjp_from_residuals(
        cell_prev, gates_res, tanh_c_res, cot_cell, cot_hidden
    )
    # Return gradients for all 3 primals; cotangents for residual outputs are
    # zero so no gradient flows back through them.
    return d_inp, d_hid, d_prev


def metal_lstm_scan(x, Wx, Wh, b, precise: bool = False, differentiable: bool = False,
                    save_residuals: bool = False):
    """Full LSTM forward using fused cell kernel.
    x: (B, T, D), Wx: (4H, D), Wh: (4H, H), b: (4H,)
    Returns hidden states (B, T, H).

    precise=True uses metal::precise::{tanh, exp} (~1e-7 vs mlx.nn.LSTM).
    differentiable=True routes through mx.custom_function so mx.grad flows.
    save_residuals=True uses the residual-saving cell + VJP kernels — backward
        avoids recomputing gate activations. Modest speedup at training time;
        ignored when differentiable=False. Default False (matches default
        forward kernel for inference).
    """
    if precise is False and _env_use_precise():
        precise = True
    if differentiable:
        if save_residuals:
            def cell(i_p, h_p, c_p):
                c, h, _, _ = metal_lstm_cell_v2(i_p, h_p, c_p)
                return c, h
        else:
            cell = metal_lstm_cell
    else:
        cell = lambda i_p, h_p, c_p: _lstm_cell_step(i_p, h_p, c_p, precise=precise)
    B, T, _ = x.shape
    H = Wh.shape[-1]
    e_proj = x @ Wx.T + b
    h = mx.zeros((B, H))
    c = mx.zeros((B, H))
    outs = []
    for t in range(T):
        h_proj = h @ Wh.T
        c, h = cell(e_proj[:, t, :], h_proj, c)
        outs.append(h)
    return mx.stack(outs, axis=1)


class MetalLSTM(mlxnn.Module):
    """Drop-in replacement for mlx.nn.LSTM. Same Wx/Wh/bias weight layout.

    Set `differentiable=True` (default for training mode) to route through
    the VJP-wired Metal kernel so `mx.grad(model)` works. Inference-only
    paths can set it False for a marginally lower-overhead call.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 precise: bool = False, differentiable: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.precise = precise
        self.differentiable = differentiable
        scale = (1.0 / hidden_size) ** 0.5
        self.Wx = mx.random.uniform(-scale, scale, (4 * hidden_size, input_size))
        self.Wh = mx.random.uniform(-scale, scale, (4 * hidden_size, hidden_size))
        self.bias = mx.zeros((4 * hidden_size,)) if bias else None

    def __call__(self, x):
        bias = self.bias if self.bias is not None else mx.zeros((4 * self.hidden_size,))
        h = metal_lstm_scan(x, self.Wx, self.Wh, bias,
                            precise=self.precise,
                            differentiable=self.differentiable)
        return h, None


# ============================================================================
# Full-scan kernel — one launch for the entire T-step LSTM scan
# ============================================================================
# Eliminates ~84 sequential kernel launches (per-timestep cell + h @ Wh matmul)
# by running everything in a single Metal launch. Threadgroup-cooperative:
# - One threadgroup per batch element, 4H threads each
# - Each thread owns ONE gate-column for all timesteps
# - h_shared and gate_shared in threadgroup memory carry state across timesteps
#
# Restriction: currently hardcoded H <= 64 (threadgroup-shared array sizes).
# Falls back to the per-cell scan if H > 64.
#
# Speed (vs metal cell-scan, Apple M4 Pro, T=42, H=64):
#   batch=1     0.94ms → 0.47ms  (2.0× faster)
#   batch=16    1.24ms → 0.55ms  (2.3× faster)
#   batch=64    1.14ms → 0.70ms  (1.6× faster)
#   batch=256   1.12ms → 1.42ms  (0.79× — slower, naive matmul becomes the bottleneck)
#   batch=512   1.48ms → 2.72ms  (0.54× — much slower)
#
# At larger batches the naive per-thread recurrent matmul is bandwidth-bound;
# simdgroup_matrix tiling would fix this but is a separate engineering effort.
# For now, use auto-select in MetalLSTMFast (full-scan for small batches).

_FULL_SCAN_SOURCE = """
    uint batch_idx = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint B = e_proj_shape[0];
    uint T = e_proj_shape[1];
    uint H = Wh_shape[1];
    uint fourH = 4u * H;

    if (batch_idx >= B) return;
    if (tid >= fourH) return;

    uint gate_id = tid / H;
    uint h_idx = tid % H;

    threadgroup float h_shared[64];
    threadgroup float gate_shared[256];

    float c_self = 0.0f;

    if (tid < H) {
        h_shared[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < T; t++) {
        float acc = 0.0f;
        device const float* wh_row = Wh + tid * H;
        for (uint k = 0; k < H; k++) {
            acc += h_shared[k] * wh_row[k];
        }
        acc += e_proj[batch_idx * T * fourH + t * fourH + tid];

        float gate = (gate_id == 2u) ? metal::fast::tanh(acc) : fast_sigmoid_mlxa(acc);
        gate_shared[tid] = gate;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < H) {
            float i_g = gate_shared[h_idx];
            float f_g = gate_shared[H + h_idx];
            float g_g = gate_shared[2u * H + h_idx];
            float o_g = gate_shared[3u * H + h_idx];
            float c_new = f_g * c_self + i_g * g_g;
            float h_new = o_g * metal::fast::tanh(c_new);
            c_self = c_new;
            h_shared[h_idx] = h_new;
            h_out[batch_idx * T * H + t * H + h_idx] = h_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

_FULL_SCAN_KERNEL = mx.fast.metal_kernel(
    name="mlxa_lstm_full_scan",
    input_names=["e_proj", "Wh"],
    output_names=["h_out"],
    source=_FULL_SCAN_SOURCE,
    header=_HEADER,
    ensure_row_contiguous=True,
)


def metal_lstm_full_scan(x, Wx, Wh, b):
    """One-launch full LSTM scan. Hidden-size limited to 64 by hardcoded
    threadgroup arrays — falls back to cell-scan otherwise."""
    B, T, _ = x.shape
    H = Wh.shape[-1]
    if H > 64:
        return metal_lstm_scan(x, Wx, Wh, b)
    fourH = 4 * H
    e_proj = x @ Wx.T + b
    return _FULL_SCAN_KERNEL(
        inputs=[e_proj, Wh],
        output_shapes=[(B, T, H)],
        output_dtypes=[mx.float32],
        grid=(B * fourH, 1, 1),
        threadgroup=(fourH, 1, 1),
    )[0]


def metal_lstm_scan_auto(x, Wx, Wh, b, batch_cutoff: int = 128):
    """Auto-select: full-scan for B < cutoff (kernel-launch bound), per-cell
    scan for B >= cutoff (compute bound, where the per-cell + MLX matmul is
    better than the naive matmul inside our full-scan kernel).

    Inference path only (no VJP). For training, use MetalLSTM(differentiable=True)
    which routes through metal_lstm_scan with the VJP cell kernel.
    """
    B = x.shape[0]
    H = Wh.shape[-1]
    if H <= 64 and B < batch_cutoff:
        return metal_lstm_full_scan(x, Wx, Wh, b)
    return metal_lstm_scan(x, Wx, Wh, b)


# ============================================================================
# Grouped LSTM cell kernel — G branches in one launch
# ============================================================================

def _make_grouped_kernel_source(precise: bool) -> str:
    sigmoid4 = "stable_sigmoid4_mlxa" if precise else "fast_sigmoid4_mlxa"
    sigmoid = "stable_sigmoid_mlxa" if precise else "fast_sigmoid_mlxa"
    tanh = "metal::precise::tanh" if precise else "metal::fast::tanh"
    return f"""
    uint hidden_size = cell_prev_shape[2];
    uint n_groups = cell_prev_shape[1];
    uint batch_size = cell_prev_shape[0];
    uint stride_4h = 4u * hidden_size;
    uint h_quads = (hidden_size + 3u) / 4u;
    uint per_batch = n_groups * h_quads;
    uint total = batch_size * per_batch;

    uint idx = thread_position_in_grid.x;
    if (idx >= total) return;

    uint batch_idx = idx / per_batch;
    uint within = idx % per_batch;
    uint group_idx = within / h_quads;
    uint h_base = (within % h_quads) * 4u;

    uint base = (batch_idx * n_groups + group_idx) * stride_4h + h_base;
    uint prev_base = (batch_idx * n_groups + group_idx) * hidden_size + h_base;

    if (h_base + 4u <= hidden_size) {{
        float4 i4 = {sigmoid4}(
            *reinterpret_cast<const device float4*>(input_proj + base) +
            *reinterpret_cast<const device float4*>(hidden_proj + base));
        float4 f4 = {sigmoid4}(
            *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + hidden_size));
        float4 g4 = {tanh}(
            *reinterpret_cast<const device float4*>(input_proj + base + 2u * hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + 2u * hidden_size));
        float4 o4 = {sigmoid4}(
            *reinterpret_cast<const device float4*>(input_proj + base + 3u * hidden_size) +
            *reinterpret_cast<const device float4*>(hidden_proj + base + 3u * hidden_size));
        float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);
        float4 c_new4 = f4 * c_prev4 + i4 * g4;
        *reinterpret_cast<device float4*>(output_cell + prev_base) = c_new4;
        *reinterpret_cast<device float4*>(output_hidden + prev_base) =
            o4 * {tanh}(c_new4);
        return;
    }}

    for (uint k = 0u; k < 4u && (h_base + k) < hidden_size; k++) {{
        uint b = base + k;
        uint pb = prev_base + k;
        float i_g = {sigmoid}(input_proj[b] + hidden_proj[b]);
        float f_g = {sigmoid}(input_proj[b + hidden_size] + hidden_proj[b + hidden_size]);
        float g_g = {tanh}(input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size]);
        float o_g = {sigmoid}(input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size]);
        float c_prev = cell_prev[pb];
        float c_new = f_g * c_prev + i_g * g_g;
        output_cell[pb] = c_new;
        output_hidden[pb] = o_g * {tanh}(c_new);
    }}
"""


_GROUPED_CELL_KERNEL_FAST = mx.fast.metal_kernel(
    name="mlxa_grouped_lstm_cell_fast",
    input_names=["input_proj", "hidden_proj", "cell_prev"],
    output_names=["output_cell", "output_hidden"],
    source=_make_grouped_kernel_source(precise=False),
    header=_HEADER,
    ensure_row_contiguous=True,
)

_GROUPED_CELL_KERNEL_PRECISE = mx.fast.metal_kernel(
    name="mlxa_grouped_lstm_cell_precise",
    input_names=["input_proj", "hidden_proj", "cell_prev"],
    output_names=["output_cell", "output_hidden"],
    source=_make_grouped_kernel_source(precise=True),
    header=_HEADER,
    ensure_row_contiguous=True,
)


def _grouped_lstm_cell_step(input_proj, hidden_proj, cell_prev, precise: bool = False):
    B, G, fourH = input_proj.shape
    H = fourH // 4
    h_quads = (H + 3) // 4
    total = B * G * h_quads
    # For grouped kernel, "effective batch" for sizing is B*G
    tg = _pick_threads_per_group(H, B * G, total)
    kernel = _GROUPED_CELL_KERNEL_PRECISE if precise else _GROUPED_CELL_KERNEL_FAST
    return kernel(
        inputs=[input_proj, hidden_proj, cell_prev],
        output_shapes=[(B, G, H), (B, G, H)],
        output_dtypes=[mx.float32, mx.float32],
        grid=(total, 1, 1),
        threadgroup=(tg, 1, 1),
    )


# ============================================================================
# Grouped VJP kernel — backward pass for the grouped LSTM cell
# ============================================================================

def _make_grouped_vjp_kernel_source(precise: bool) -> str:
    sigmoid4 = "stable_sigmoid4_mlxa" if precise else "fast_sigmoid4_mlxa"
    sigmoid = "stable_sigmoid_mlxa" if precise else "fast_sigmoid_mlxa"
    tanh = "metal::precise::tanh" if precise else "metal::fast::tanh"
    return f"""
    uint hidden_size = cell_prev_shape[2];
    uint n_groups = cell_prev_shape[1];
    uint batch_size = cell_prev_shape[0];
    uint stride_4h = 4u * hidden_size;
    uint h_quads = (hidden_size + 3u) / 4u;
    uint per_batch = n_groups * h_quads;
    uint total = batch_size * per_batch;
    uint idx = thread_position_in_grid.x;
    if (idx >= total) return;
    uint batch_idx = idx / per_batch;
    uint within = idx % per_batch;
    uint group_idx = within / h_quads;
    uint h_base = (within % h_quads) * 4u;
    uint base = (batch_idx * n_groups + group_idx) * stride_4h + h_base;
    uint prev_base = (batch_idx * n_groups + group_idx) * hidden_size + h_base;

    if (h_base + 4u <= hidden_size) {{
        float4 gi4 = *reinterpret_cast<const device float4*>(input_proj + base) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base);
        float4 gf4 = *reinterpret_cast<const device float4*>(input_proj + base + hidden_size) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base + hidden_size);
        float4 gg4 = *reinterpret_cast<const device float4*>(input_proj + base + 2u * hidden_size) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base + 2u * hidden_size);
        float4 go4 = *reinterpret_cast<const device float4*>(input_proj + base + 3u * hidden_size) +
                     *reinterpret_cast<const device float4*>(hidden_proj + base + 3u * hidden_size);

        float4 i4 = {sigmoid4}(gi4);
        float4 f4 = {sigmoid4}(gf4);
        float4 g4 = {tanh}(gg4);
        float4 o4 = {sigmoid4}(go4);

        float4 c_prev4 = *reinterpret_cast<const device float4*>(cell_prev + prev_base);
        float4 c_new4 = f4 * c_prev4 + i4 * g4;
        float4 tanh_c4 = {tanh}(c_new4);

        float4 cot_c4 = *reinterpret_cast<const device float4*>(cot_cell + prev_base);
        float4 cot_h4 = *reinterpret_cast<const device float4*>(cot_hidden + prev_base);

        float4 dc4 = cot_c4 + cot_h4 * o4 * (1.0f - tanh_c4 * tanh_c4);
        float4 do4 = cot_h4 * tanh_c4;
        float4 di4 = dc4 * g4;
        float4 df4 = dc4 * c_prev4;
        float4 dg4 = dc4 * i4;

        float4 d_i_gate4 = di4 * i4 * (1.0f - i4);
        float4 d_f_gate4 = df4 * f4 * (1.0f - f4);
        float4 d_g_gate4 = dg4 * (1.0f - g4 * g4);
        float4 d_o_gate4 = do4 * o4 * (1.0f - o4);

        *reinterpret_cast<device float4*>(d_input_proj + base) = d_i_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + hidden_size) = d_f_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + 2u * hidden_size) = d_g_gate4;
        *reinterpret_cast<device float4*>(d_input_proj + base + 3u * hidden_size) = d_o_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base) = d_i_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + hidden_size) = d_f_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + 2u * hidden_size) = d_g_gate4;
        *reinterpret_cast<device float4*>(d_hidden_proj + base + 3u * hidden_size) = d_o_gate4;

        *reinterpret_cast<device float4*>(d_cell_prev + prev_base) = dc4 * f4;
        return;
    }}

    for (uint k = 0u; k < 4u && (h_base + k) < hidden_size; k++) {{
        uint b = base + k;
        uint pb = prev_base + k;
        float gi = input_proj[b] + hidden_proj[b];
        float gf = input_proj[b + hidden_size] + hidden_proj[b + hidden_size];
        float gg = input_proj[b + 2u * hidden_size] + hidden_proj[b + 2u * hidden_size];
        float go = input_proj[b + 3u * hidden_size] + hidden_proj[b + 3u * hidden_size];
        float i_g = {sigmoid}(gi);
        float f_g = {sigmoid}(gf);
        float g_g = {tanh}(gg);
        float o_g = {sigmoid}(go);
        float c_prev_v = cell_prev[pb];
        float c_new = f_g * c_prev_v + i_g * g_g;
        float tanh_c = {tanh}(c_new);
        float cot_c_v = cot_cell[pb];
        float cot_h_v = cot_hidden[pb];
        float dc = cot_c_v + cot_h_v * o_g * (1.0f - tanh_c * tanh_c);
        float do_v = cot_h_v * tanh_c;
        float di_v = dc * g_g;
        float df_v = dc * c_prev_v;
        float dg_v = dc * i_g;
        float d_i_gate = di_v * i_g * (1.0f - i_g);
        float d_f_gate = df_v * f_g * (1.0f - f_g);
        float d_g_gate = dg_v * (1.0f - g_g * g_g);
        float d_o_gate = do_v * o_g * (1.0f - o_g);
        d_input_proj[b] = d_i_gate;
        d_input_proj[b + hidden_size] = d_f_gate;
        d_input_proj[b + 2u * hidden_size] = d_g_gate;
        d_input_proj[b + 3u * hidden_size] = d_o_gate;
        d_hidden_proj[b] = d_i_gate;
        d_hidden_proj[b + hidden_size] = d_f_gate;
        d_hidden_proj[b + 2u * hidden_size] = d_g_gate;
        d_hidden_proj[b + 3u * hidden_size] = d_o_gate;
        d_cell_prev[pb] = dc * f_g;
    }}
"""


_GROUPED_CELL_VJP_KERNEL_FAST = mx.fast.metal_kernel(
    name="mlxa_grouped_lstm_cell_vjp_fast",
    input_names=["input_proj", "hidden_proj", "cell_prev", "cot_cell", "cot_hidden"],
    output_names=["d_input_proj", "d_hidden_proj", "d_cell_prev"],
    source=_make_grouped_vjp_kernel_source(precise=False),
    header=_HEADER,
    ensure_row_contiguous=True,
)


def _grouped_lstm_cell_vjp_step(input_proj, hidden_proj, cell_prev, cot_cell, cot_hidden):
    B, G, fourH = input_proj.shape
    H = fourH // 4
    h_quads = (H + 3) // 4
    total = B * G * h_quads
    tg = _pick_threads_per_group(H, B * G, total)
    return _GROUPED_CELL_VJP_KERNEL_FAST(
        inputs=[input_proj, hidden_proj, cell_prev, cot_cell, cot_hidden],
        output_shapes=[(B, G, 4 * H), (B, G, 4 * H), (B, G, H)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
        grid=(total, 1, 1),
        threadgroup=(tg, 1, 1),
    )


@mx.custom_function
def grouped_metal_lstm_cell(input_proj, hidden_proj, cell_prev):
    """G-grouped LSTM cell with autograd-aware Metal kernels (fwd + VJP)."""
    return _grouped_lstm_cell_step(input_proj, hidden_proj, cell_prev, precise=False)


@grouped_metal_lstm_cell.vjp
def _grouped_metal_lstm_cell_vjp(primals, cotangents, outputs):
    input_proj, hidden_proj, cell_prev = primals
    cot_cell, cot_hidden = cotangents
    return _grouped_lstm_cell_vjp_step(input_proj, hidden_proj, cell_prev, cot_cell, cot_hidden)


def metal_grouped_lstm_scan(x, Wx_stacked, Wh_stacked, b_stacked,
                            precise: bool = False, differentiable: bool = False):
    """G parallel LSTMs sharing the same input.
    x: (B, T, D_in), Wx_stacked: (G, 4H, D_in), Wh_stacked: (G, 4H, H), b_stacked: (G, 4H)
    Returns: (B, T, G, H).

    differentiable=True routes through mx.custom_function so mx.grad flows
    (uses the grouped VJP kernel). Required for training.
    """
    if precise is False and _env_use_precise():
        precise = True
    B, T, D_in = x.shape
    G, _, H = Wh_stacked.shape

    Wx_flat = Wx_stacked.reshape(-1, D_in)
    e_proj = (x @ Wx_flat.T).reshape(B, T, G, 4 * H) + b_stacked

    cell = grouped_metal_lstm_cell if differentiable else (
        lambda i_p, h_p, c_p: _grouped_lstm_cell_step(i_p, h_p, c_p, precise=precise)
    )

    h = mx.zeros((B, G, H))
    c = mx.zeros((B, G, H))
    outs = []
    for t in range(T):
        h_proj = mx.einsum("bgh,gih->bgi", h, Wh_stacked)
        c, h = cell(e_proj[:, t, :, :], h_proj, c)
        outs.append(h)
    return mx.stack(outs, axis=1)


class GroupedMetalLSTM(mlxnn.Module):
    """G parallel LSTMs (same input, different weights), one kernel launch per timestep.
    differentiable=True (default) enables training via the grouped VJP kernel."""

    def __init__(self, input_size: int, hidden_size: int, n_groups: int,
                 bias: bool = True, precise: bool = False, differentiable: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.precise = precise
        self.differentiable = differentiable
        scale = (1.0 / hidden_size) ** 0.5
        self.Wx = mx.random.uniform(-scale, scale, (n_groups, 4 * hidden_size, input_size))
        self.Wh = mx.random.uniform(-scale, scale, (n_groups, 4 * hidden_size, hidden_size))
        self.bias = mx.zeros((n_groups, 4 * hidden_size)) if bias else None

    def __call__(self, x, return_last_only: bool = False):
        bias = self.bias if self.bias is not None else mx.zeros((self.n_groups, 4 * self.hidden_size))
        out = metal_grouped_lstm_scan(x, self.Wx, self.Wh, bias,
                                       precise=self.precise,
                                       differentiable=self.differentiable)
        return out[:, -1, :, :] if return_last_only else out
