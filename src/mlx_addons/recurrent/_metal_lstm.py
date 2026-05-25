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


def metal_lstm_scan(x, Wx, Wh, b, precise: bool = False):
    """Full LSTM forward using fused cell kernel.
    x: (B, T, D), Wx: (4H, D), Wh: (4H, H), b: (4H,)
    Returns hidden states (B, T, H).
    precise=True uses metal::precise::{tanh, exp} (~1e-7 vs mlx.nn.LSTM,
    slightly slower); fast=False uses fast::tanh and fast_sigmoid (~3e-5 diff,
    fastest).
    """
    if precise is False and _env_use_precise():
        precise = True
    B, T, _ = x.shape
    H = Wh.shape[-1]
    e_proj = x @ Wx.T + b
    h = mx.zeros((B, H))
    c = mx.zeros((B, H))
    outs = []
    for t in range(T):
        h_proj = h @ Wh.T
        c, h = _lstm_cell_step(e_proj[:, t, :], h_proj, c, precise=precise)
        outs.append(h)
    return mx.stack(outs, axis=1)


class MetalLSTM(mlxnn.Module):
    """Drop-in replacement for mlx.nn.LSTM. Same Wx/Wh/bias weight layout."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 precise: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.precise = precise
        scale = (1.0 / hidden_size) ** 0.5
        self.Wx = mx.random.uniform(-scale, scale, (4 * hidden_size, input_size))
        self.Wh = mx.random.uniform(-scale, scale, (4 * hidden_size, hidden_size))
        self.bias = mx.zeros((4 * hidden_size,)) if bias else None

    def __call__(self, x):
        bias = self.bias if self.bias is not None else mx.zeros((4 * self.hidden_size,))
        h = metal_lstm_scan(x, self.Wx, self.Wh, bias, precise=self.precise)
        return h, None


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


def metal_grouped_lstm_scan(x, Wx_stacked, Wh_stacked, b_stacked, precise: bool = False):
    """G parallel LSTMs sharing the same input.
    x: (B, T, D_in), Wx_stacked: (G, 4H, D_in), Wh_stacked: (G, 4H, H), b_stacked: (G, 4H)
    Returns: (B, T, G, H).
    """
    if precise is False and _env_use_precise():
        precise = True
    B, T, D_in = x.shape
    G, _, H = Wh_stacked.shape

    Wx_flat = Wx_stacked.reshape(-1, D_in)
    e_proj = (x @ Wx_flat.T).reshape(B, T, G, 4 * H) + b_stacked

    h = mx.zeros((B, G, H))
    c = mx.zeros((B, G, H))
    outs = []
    for t in range(T):
        h_proj = mx.einsum("bgh,gih->bgi", h, Wh_stacked)
        c, h = _grouped_lstm_cell_step(e_proj[:, t, :, :], h_proj, c, precise=precise)
        outs.append(h)
    return mx.stack(outs, axis=1)


class GroupedMetalLSTM(mlxnn.Module):
    """G parallel LSTMs (same input, different weights), one kernel launch per timestep."""

    def __init__(self, input_size: int, hidden_size: int, n_groups: int,
                 bias: bool = True, precise: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_groups = n_groups
        self.precise = precise
        scale = (1.0 / hidden_size) ** 0.5
        self.Wx = mx.random.uniform(-scale, scale, (n_groups, 4 * hidden_size, input_size))
        self.Wh = mx.random.uniform(-scale, scale, (n_groups, 4 * hidden_size, hidden_size))
        self.bias = mx.zeros((n_groups, 4 * hidden_size)) if bias else None

    def __call__(self, x, return_last_only: bool = False):
        bias = self.bias if self.bias is not None else mx.zeros((self.n_groups, 4 * self.hidden_size))
        out = metal_grouped_lstm_scan(x, self.Wx, self.Wh, bias, precise=self.precise)
        return out[:, -1, :, :] if return_last_only else out
