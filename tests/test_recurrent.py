"""Tests for mlx_addons.recurrent — Metal LSTM cell + VJP correctness."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

mlxnn = pytest.importorskip("mlx.nn")

from mlx_addons.recurrent import (
    GroupedMetalLSTM,
    MetalLSTM,
    metal_grouped_lstm_scan,
    metal_lstm_scan,
)
from mlx_addons.recurrent._metal_lstm import metal_lstm_cell


# ---------------- Forward correctness ----------------

class TestForwardCorrectness:
    def test_metal_matches_mlx_nn_lstm(self):
        """MetalLSTM output should match mlx.nn.LSTM bit-equivalently when
        given the same weights."""
        H, D, B, T = 64, 27, 4, 42
        rng = np.random.RandomState(0)
        ref = mlxnn.LSTM(D, H)
        x = mx.array(rng.randn(B, T, D).astype(np.float32))
        h_ref, _ = ref(x); mx.eval(h_ref)
        # Use the same weights via metal_lstm_scan
        h_metal = metal_lstm_scan(x, ref.Wx, ref.Wh, ref.bias)
        mx.eval(h_metal)
        diff = float(mx.max(mx.abs(h_ref - h_metal)))
        assert diff < 1e-5, f"fast max_diff {diff}"

    def test_precise_tighter_than_fast(self):
        """Precise math should match mlx.nn.LSTM as tightly or tighter than fast."""
        H, D, B, T = 64, 27, 4, 42
        rng = np.random.RandomState(0)
        ref = mlxnn.LSTM(D, H)
        x = mx.array(rng.randn(B, T, D).astype(np.float32))
        h_ref, _ = ref(x); mx.eval(h_ref)
        h_fast = metal_lstm_scan(x, ref.Wx, ref.Wh, ref.bias, precise=False)
        h_prec = metal_lstm_scan(x, ref.Wx, ref.Wh, ref.bias, precise=True)
        mx.eval(h_fast, h_prec)
        d_fast = float(mx.max(mx.abs(h_ref - h_fast)))
        d_prec = float(mx.max(mx.abs(h_ref - h_prec)))
        assert d_prec <= d_fast + 1e-7, f"precise {d_prec} not <= fast {d_fast}"


# ---------------- VJP / backward correctness ----------------

def _ref_cell(input_proj, hidden_proj, cell_prev):
    """Pure-MLX reference LSTM cell — autograd works through this natively."""
    gates = input_proj + hidden_proj
    H = cell_prev.shape[-1]
    i = mx.sigmoid(gates[:, :H])
    f = mx.sigmoid(gates[:, H:2 * H])
    g = mx.tanh(gates[:, 2 * H:3 * H])
    o = mx.sigmoid(gates[:, 3 * H:4 * H])
    c_new = f * cell_prev + i * g
    h_new = o * mx.tanh(c_new)
    return c_new, h_new


def _ref_scan(x, Wx, Wh, b):
    B, T, _ = x.shape
    H = Wh.shape[-1]
    e_proj = x @ Wx.T + b
    h = mx.zeros((B, H))
    c = mx.zeros((B, H))
    outs = []
    for t in range(T):
        h_proj = h @ Wh.T
        c, h = _ref_cell(e_proj[:, t, :], h_proj, c)
        outs.append(h)
    return mx.stack(outs, axis=1)


class TestVJPCorrectness:
    def test_cell_gradients_match_reference(self):
        """metal_lstm_cell's VJP should match mlx.grad through the pure-MLX
        reference cell, to within float32 epsilon."""
        H, B = 64, 4
        rng = np.random.RandomState(0)
        ip = mx.array(rng.randn(B, 4 * H).astype(np.float32))
        hp = mx.array(rng.randn(B, 4 * H).astype(np.float32))
        cp = mx.array(rng.randn(B, H).astype(np.float32))

        def loss_ref(ip, hp, cp):
            c, h = _ref_cell(ip, hp, cp)
            return mx.sum(c) + 2 * mx.sum(h)

        def loss_metal(ip, hp, cp):
            c, h = metal_lstm_cell(ip, hp, cp)
            return mx.sum(c) + 2 * mx.sum(h)

        g_ref = mx.grad(loss_ref, argnums=(0, 1, 2))(ip, hp, cp)
        g_metal = mx.grad(loss_metal, argnums=(0, 1, 2))(ip, hp, cp)
        for gr, gm in zip(g_ref, g_metal):
            diff = float(mx.max(mx.abs(gr - gm)))
            norm = float(mx.max(mx.abs(gr))) + 1e-9
            assert diff / norm < 1e-5, f"rel_err {diff/norm:.3e}"

    def test_full_scan_gradients_match(self):
        """The full T-step scan with VJP kernel should match autograd through
        the pure-MLX reference scan for gradients of (x, Wx, Wh, b)."""
        H, D, B, T = 64, 27, 4, 10
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(B, T, D).astype(np.float32))
        Wx = mx.array(rng.randn(4 * H, D).astype(np.float32) * 0.1)
        Wh = mx.array(rng.randn(4 * H, H).astype(np.float32) * 0.1)
        b = mx.array(rng.randn(4 * H).astype(np.float32) * 0.01)

        def loss_ref(x, Wx, Wh, b):
            return mx.sum(_ref_scan(x, Wx, Wh, b) ** 2)

        def loss_metal(x, Wx, Wh, b):
            return mx.sum(metal_lstm_scan(x, Wx, Wh, b, differentiable=True) ** 2)

        g_ref = mx.grad(loss_ref, argnums=(0, 1, 2, 3))(x, Wx, Wh, b)
        g_metal = mx.grad(loss_metal, argnums=(0, 1, 2, 3))(x, Wx, Wh, b)
        for name, gr, gm in zip(("dx", "dWx", "dWh", "db"), g_ref, g_metal):
            diff = float(mx.max(mx.abs(gr - gm)))
            norm = float(mx.max(mx.abs(gr))) + 1e-9
            assert diff / norm < 1e-4, f"{name} rel_err {diff/norm:.3e}"


class TestMetalLSTMTrainable:
    def test_metal_lstm_module_grad_flow(self):
        """MetalLSTM as a Module should produce gradients via mx.grad."""
        D, H, B, T = 27, 64, 4, 10
        rng = np.random.RandomState(0)
        m = MetalLSTM(D, H, differentiable=True)
        x = mx.array(rng.randn(B, T, D).astype(np.float32))

        def loss_fn(params, x):
            m.update(params)
            h, _ = m(x)
            return mx.sum(h ** 2)

        params = m.trainable_parameters()
        grads = mx.grad(loss_fn)(params, x)
        # Should produce non-zero gradients for Wx, Wh, bias
        assert "Wx" in grads
        assert "Wh" in grads
        assert "bias" in grads
        assert float(mx.max(mx.abs(grads["Wx"]))) > 0
        assert float(mx.max(mx.abs(grads["Wh"]))) > 0


# ---------------- Grouped LSTM (forward only — no VJP yet) ----------------

class TestGroupedForward:
    def test_grouped_shape(self):
        D, H, B, T, G = 27, 64, 4, 10, 4
        gl = GroupedMetalLSTM(D, H, G)
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(B, T, D).astype(np.float32))
        out = gl(x)
        mx.eval(out)
        assert out.shape == (B, T, G, H)

    def test_grouped_no_nan(self):
        D, H, B, T, G = 27, 64, 4, 10, 4
        gl = GroupedMetalLSTM(D, H, G)
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(B, T, D).astype(np.float32))
        out = np.array(gl(x))
        mx.eval(out)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()


class TestGroupedVJP:
    """metal_grouped_lstm_scan(differentiable=True) gradients must match
    autograd through a pure-MLX reference."""

    def test_grouped_scan_gradients_match(self):
        H, D, B, T, G = 64, 27, 4, 10, 4
        rng = np.random.RandomState(0)
        x = mx.array(rng.randn(B, T, D).astype(np.float32))
        Wx = mx.array(rng.randn(G, 4 * H, D).astype(np.float32) * 0.1)
        Wh = mx.array(rng.randn(G, 4 * H, H).astype(np.float32) * 0.1)
        b = mx.array(rng.randn(G, 4 * H).astype(np.float32) * 0.01)

        def ref_grouped(x, Wx_s, Wh_s, b_s):
            B_, T_, D_ = x.shape
            G_, _, H_ = Wh_s.shape
            Wx_flat = Wx_s.reshape(-1, D_)
            e_proj = (x @ Wx_flat.T).reshape(B_, T_, G_, 4 * H_) + b_s
            h = mx.zeros((B_, G_, H_)); c = mx.zeros((B_, G_, H_))
            outs = []
            for t in range(T_):
                h_proj = mx.einsum("bgh,gih->bgi", h, Wh_s)
                gates = e_proj[:, t, :, :] + h_proj
                i = mx.sigmoid(gates[..., :H_])
                f = mx.sigmoid(gates[..., H_:2 * H_])
                g_ = mx.tanh(gates[..., 2 * H_:3 * H_])
                o = mx.sigmoid(gates[..., 3 * H_:4 * H_])
                c = f * c + i * g_
                h = o * mx.tanh(c)
                outs.append(h)
            return mx.stack(outs, axis=1)

        def loss_ref(x, Wx, Wh, b):
            return mx.sum(ref_grouped(x, Wx, Wh, b) ** 2)

        def loss_metal(x, Wx, Wh, b):
            return mx.sum(metal_grouped_lstm_scan(x, Wx, Wh, b, differentiable=True) ** 2)

        g_ref = mx.grad(loss_ref, argnums=(0, 1, 2, 3))(x, Wx, Wh, b)
        g_metal = mx.grad(loss_metal, argnums=(0, 1, 2, 3))(x, Wx, Wh, b)
        for name, gr, gm in zip(("dx", "dWx", "dWh", "db"), g_ref, g_metal):
            diff = float(mx.max(mx.abs(gr - gm)))
            norm = float(mx.max(mx.abs(gr))) + 1e-9
            assert diff / norm < 1e-4, f"{name} rel_err {diff/norm:.3e}"
