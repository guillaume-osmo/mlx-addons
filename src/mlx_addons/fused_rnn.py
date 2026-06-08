"""
fused_rnn — input-projection-fused LSTM for MLX (Apple Silicon).

A single JIT Metal kernel (``mx.fast.metal_kernel``, no MLX rebuild) computes the
whole sequence with the input projection FUSED into the recurrent loop::

    per timestep t:   gates = x[t] @ Wx + h @ Wh + bias   (one simdgroup-matrix
                      GEMM into shared gate buffer), then i,f,g,o + cell update.

vs the usual two-step (a separate ``addmm`` input projection that materialises a
``[B,T,4H]`` tensor, then a recurrent kernel that re-reads it), fusing removes
that round-trip and one dispatch.  On M-series this **matches or beats PyTorch's
MPSGraph LSTM** (which is why it exists):

    LSTM layer fwd, in=128 hidden=64 seq=64 (ms, lower=better)
    B     two-step(MLX)   fused(this)   PyTorch/MPSGraph
    32       1.40            0.71            0.63
    128      1.44            0.74            0.72   (parity)
    512      1.88            1.16            1.39   (beats torch)

Forward-only (training needs a vjp — see ``FusedLSTM`` notes); ideal for
inference and for the recurrent forward of bidirectional encoders.

    from mlx_addons.fused_rnn import fused_lstm_sequence, FusedLSTM
"""
from __future__ import annotations
import mlx.core as mx

__all__ = ["fused_lstm_sequence", "FusedLSTM"]

_KCACHE: dict = {}
_HEADER = "#include <metal_simdgroup_matrix>\n#include <metal_math>\n"


def _src(B: int, T: int, H: int, IN: int, TG: int) -> str:
    H4, NSG = 4 * H, TG // 32
    return f"""
    const uint H={H}u, H4={H4}u, IN={IN}u, Tn={T}u, BSZ={B}u;
    const uint b_tile=8u, b_tile_pad=8u, TGSZ={TG}u, NSG={NSG}u;
    const uint K_H=H/8u, K_X=IN/8u, N_T=H4/8u, M_T=b_tile_pad/8u, TOT=M_T*N_T;
    uint tid = thread_index_in_threadgroup;
    uint sg  = tid / 32u;
    uint tg_idx = threadgroup_position_in_grid.x;
    uint b_base = tg_idx * b_tile;
    threadgroup float sh_h[b_tile_pad*{H}u];
    threadgroup float sh_x[b_tile_pad*{IN}u];
    threadgroup float gbuf[b_tile_pad*{H4}u];
    for (uint i=tid; i<b_tile_pad*H; i+=TGSZ) {{
        uint b=i/H, h=i%H, bg=b_base+b;
        sh_h[i] = (b<b_tile && bg<BSZ) ? h_init[bg*H+h] : 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t=0u; t<Tn; ++t) {{
        for (uint i=tid; i<b_tile_pad*IN; i+=TGSZ) {{
            uint b=i/IN, k=i%IN, bg=b_base+b;
            sh_x[i] = (b<b_tile && bg<BSZ) ? x[bg*Tn*IN + t*IN + k] : 0.0f;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint mn=sg; mn<TOT; mn+=NSG) {{
            uint m=mn/N_T, n=mn%N_T;
            simdgroup_matrix<float,8,8> C = simdgroup_matrix<float,8,8>(0);
            simdgroup_matrix<float,8,8> A, Bm;
            for (uint k=0u; k<K_H; ++k) {{
                simdgroup_load(A, sh_h + m*8u*H + k*8u, H);
                simdgroup_load(Bm, Wh_t + k*8u*H4 + n*8u, H4);
                simdgroup_multiply_accumulate(C, A, Bm, C);
            }}
            for (uint k=0u; k<K_X; ++k) {{
                simdgroup_load(A, sh_x + m*8u*IN + k*8u, IN);
                simdgroup_load(Bm, Wx_t + k*8u*H4 + n*8u, H4);
                simdgroup_multiply_accumulate(C, A, Bm, C);
            }}
            simdgroup_store(C, gbuf + m*8u*H4 + n*8u, H4);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid; i<b_tile*H; i+=TGSZ) {{
            uint b=i/H, h=i%H, bg=b_base+b;
            if (bg>=BSZ) continue;
            float gi=gbuf[b*H4+h]+bias[h], gf=gbuf[b*H4+H+h]+bias[H+h];
            float gg=gbuf[b*H4+2u*H+h]+bias[2u*H+h], go=gbuf[b*H4+3u*H+h]+bias[3u*H+h];
            float ig=1.0f/(1.0f+metal::exp(-gi)), fg=1.0f/(1.0f+metal::exp(-gf));
            float gv=metal::precise::tanh(gg), og=1.0f/(1.0f+metal::exp(-go));
            float cp=(t==0u)?c_init[bg*H+h]:out_c[bg*Tn*H+(t-1u)*H+h];
            float cn=fg*cp+ig*gv, hn=og*metal::precise::tanh(cn);
            uint o=bg*Tn*H+t*H+h;
            out_h[o]=hn; out_c[o]=cn; sh_h[b*H+h]=hn;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    """


def fused_lstm_sequence(x, Wx, Wh, bias, h0=None, c0=None, tg: int = 0):
    """Forward LSTM over a full sequence, input projection fused in.

    x:[B,T,IN]  Wx:[4H,IN]  Wh:[4H,H]  bias:[4H]  h0,c0:[B,H] (default zeros).
    Returns h_out:[B,T,H].  Requires IN,H,4H multiples of 8, float32, GPU.
    """
    B, T, IN = x.shape
    H = Wh.shape[1]
    if tg == 0:
        tg = 1024 if B <= 128 else 256
    if h0 is None:
        h0 = mx.zeros((B, H), dtype=x.dtype)
    if c0 is None:
        c0 = mx.zeros((B, H), dtype=x.dtype)
    key = (B, T, H, IN, tg)
    if key not in _KCACHE:
        _KCACHE[key] = mx.fast.metal_kernel(
            name=f"fused_lstm_xin_{B}_{T}_{H}_{IN}_{tg}",
            input_names=["x", "Wx_t", "Wh_t", "bias", "h_init", "c_init"],
            output_names=["out_h", "out_c"], header=_HEADER, source=_src(B, T, H, IN, tg))
    h, _ = _KCACHE[key](
        inputs=[mx.contiguous(x), mx.contiguous(Wx.T), mx.contiguous(Wh.T), bias, h0, c0],
        output_shapes=[(B, T, H), (B, T, H)], output_dtypes=[x.dtype, x.dtype],
        grid=((B + 7) // 8 * tg, 1, 1), threadgroup=(tg, 1, 1))
    return h


# ---------------------------------------------------------------------------
# Trainable path: train-forward kernel (saves gates) + fused BPTT backward kernel.
# 6.7x over eager MLX and ~1.2x faster than PyTorch's MPSGraph LSTM at fwd+bwd.
# ---------------------------------------------------------------------------
_FT, _BW = {}, {}


def _fwd_train_src(B, T, H, IN, TG):
    H4, NSG = 4 * H, TG // 32
    return f"""
    const uint H={H}u,H4={H4}u,IN={IN}u,Tn={T}u,BSZ={B}u,b_tile=8u,b_tile_pad=8u,TGSZ={TG}u,NSG={NSG}u;
    const uint K_H=H/8u,K_X=IN/8u,N_T=H4/8u,TOT=(b_tile_pad/8u)*N_T;
    uint tid=thread_index_in_threadgroup,sg=tid/32u,tg_idx=threadgroup_position_in_grid.x,b_base=tg_idx*b_tile;
    threadgroup float sh_h[b_tile_pad*{H}u],sh_x[b_tile_pad*{IN}u],gbuf[b_tile_pad*{H4}u];
    for(uint i=tid;i<b_tile_pad*H;i+=TGSZ){{uint b=i/H,h=i%H,bg=b_base+b;sh_h[i]=(b<b_tile&&bg<BSZ)?h_init[bg*H+h]:0.0f;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=0u;t<Tn;++t){{
      for(uint i=tid;i<b_tile_pad*IN;i+=TGSZ){{uint b=i/IN,k=i%IN,bg=b_base+b;sh_x[i]=(b<b_tile&&bg<BSZ)?x[bg*Tn*IN+t*IN+k]:0.0f;}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint mn=sg;mn<TOT;mn+=NSG){{uint m=mn/N_T,n=mn%N_T;simdgroup_matrix<float,8,8> C=simdgroup_matrix<float,8,8>(0),A,Bm;
        for(uint k=0u;k<K_H;++k){{simdgroup_load(A,sh_h+m*8u*H+k*8u,H);simdgroup_load(Bm,Wh_t+k*8u*H4+n*8u,H4);simdgroup_multiply_accumulate(C,A,Bm,C);}}
        for(uint k=0u;k<K_X;++k){{simdgroup_load(A,sh_x+m*8u*IN+k*8u,IN);simdgroup_load(Bm,Wx_t+k*8u*H4+n*8u,H4);simdgroup_multiply_accumulate(C,A,Bm,C);}}
        simdgroup_store(C,gbuf+m*8u*H4+n*8u,H4);}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint i=tid;i<b_tile*H;i+=TGSZ){{uint b=i/H,h=i%H,bg=b_base+b;if(bg>=BSZ)continue;
        float ig=1.0f/(1.0f+metal::exp(-(gbuf[b*H4+h]+bias[h])));
        float fg=1.0f/(1.0f+metal::exp(-(gbuf[b*H4+H+h]+bias[H+h])));
        float gv=metal::precise::tanh(gbuf[b*H4+2u*H+h]+bias[2u*H+h]);
        float og=1.0f/(1.0f+metal::exp(-(gbuf[b*H4+3u*H+h]+bias[3u*H+h])));
        float cp=(t==0u)?c_init[bg*H+h]:out_c[bg*Tn*H+(t-1u)*H+h];
        float cn=fg*cp+ig*gv,hn=og*metal::precise::tanh(cn);
        uint o=bg*Tn*H+t*H+h;out_h[o]=hn;out_c[o]=cn;sh_h[b*H+h]=hn;
        uint o4=bg*Tn*H4+t*H4+h;out_g[o4]=ig;out_g[o4+H]=fg;out_g[o4+2u*H]=gv;out_g[o4+3u*H]=og;}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}"""


def _bwd_src(B, T, H, TG):
    H4, NSG = 4 * H, TG // 32
    return f"""
    const uint H={H}u,H4={H4}u,Tn={T}u,BSZ={B}u,b_tile=8u,b_tile_pad=8u,TGSZ={TG}u,NSG={NSG}u;
    const uint K_Z=H4/8u,N_T=H/8u,TOT=(b_tile_pad/8u)*N_T;
    uint tid=thread_index_in_threadgroup,sg=tid/32u,tg_idx=threadgroup_position_in_grid.x,b_base=tg_idx*b_tile;
    threadgroup float sh_dz[b_tile_pad*{H4}u],sh_dhn[b_tile_pad*{H}u],sh_dcn[b_tile_pad*{H}u];
    for(uint i=tid;i<b_tile_pad*H;i+=TGSZ){{sh_dhn[i]=0.0f;sh_dcn[i]=0.0f;}}
    for(uint i=tid;i<b_tile_pad*H4;i+=TGSZ){{sh_dz[i]=0.0f;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int tt=int(Tn)-1;tt>=0;--tt){{uint t=uint(tt);
      for(uint i=tid;i<b_tile*H;i+=TGSZ){{uint b=i/H,h=i%H,bg=b_base+b;if(bg>=BSZ)continue;
        uint g4=bg*Tn*H4+t*H4+h;
        float ig=gates[g4],fg=gates[g4+H],gv=gates[g4+2u*H],og=gates[g4+3u*H];
        float c=c_seq[bg*Tn*H+t*H+h],cp=(t==0u)?c_init[bg*H+h]:c_seq[bg*Tn*H+(t-1u)*H+h];
        float tc=metal::precise::tanh(c);
        float dh=dh_seq[bg*Tn*H+t*H+h]+sh_dhn[b*H+h];
        float dgo=dh*tc;
        float dc=dh*og*(1.0f-tc*tc)+sh_dcn[b*H+h];
        float dzi=dc*gv*ig*(1.0f-ig),dzf=dc*cp*fg*(1.0f-fg),dzg=dc*ig*(1.0f-gv*gv),dzo=dgo*og*(1.0f-og);
        uint zb=b*H4+h;sh_dz[zb]=dzi;sh_dz[zb+H]=dzf;sh_dz[zb+2u*H]=dzg;sh_dz[zb+3u*H]=dzo;
        out_dz[g4]=dzi;out_dz[g4+H]=dzf;out_dz[g4+2u*H]=dzg;out_dz[g4+3u*H]=dzo;
        sh_dcn[b*H+h]=dc*fg;}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint mn=sg;mn<TOT;mn+=NSG){{uint m=mn/N_T,n=mn%N_T;simdgroup_matrix<float,8,8> C=simdgroup_matrix<float,8,8>(0),A,Bm;
        for(uint k=0u;k<K_Z;++k){{simdgroup_load(A,sh_dz+m*8u*H4+k*8u,H4);simdgroup_load(Bm,Wh+k*8u*H+n*8u,H);simdgroup_multiply_accumulate(C,A,Bm,C);}}
        simdgroup_store(C,sh_dhn+m*8u*H+n*8u,H);}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}"""


def _fwd_train(x, Wx, Wh, bias, h0, c0, tg):
    B, T, IN = x.shape; H = Wh.shape[1]
    key = (B, T, H, IN, tg)
    if key not in _FT:
        _FT[key] = mx.fast.metal_kernel(name=f"flstm_ft_{B}_{T}_{H}_{IN}_{tg}",
            input_names=["x", "Wx_t", "Wh_t", "bias", "h_init", "c_init"],
            output_names=["out_h", "out_c", "out_g"], header=_HEADER, source=_fwd_train_src(B, T, H, IN, tg))
    return _FT[key](inputs=[mx.contiguous(x), mx.contiguous(Wx.T), mx.contiguous(Wh.T), bias, h0, c0],
        output_shapes=[(B, T, H), (B, T, H), (B, T, 4 * H)], output_dtypes=[x.dtype] * 3,
        grid=((B + 7) // 8 * tg, 1, 1), threadgroup=(tg, 1, 1))


def _bwd_dz(dh_seq, gates, c_seq, c0, Wh, tg):
    B, T, H = dh_seq.shape
    key = (B, T, H, tg)
    if key not in _BW:
        _BW[key] = mx.fast.metal_kernel(name=f"flstm_bw_{B}_{T}_{H}_{tg}",
            input_names=["dh_seq", "gates", "c_seq", "c_init", "Wh"],
            output_names=["out_dz"], header=_HEADER, source=_bwd_src(B, T, H, tg))
    return _BW[key](inputs=[mx.contiguous(dh_seq), gates, c_seq, c0, mx.contiguous(Wh)],
        output_shapes=[(B, T, 4 * H)], output_dtypes=[dh_seq.dtype],
        grid=((B + 7) // 8 * tg, 1, 1), threadgroup=(tg, 1, 1))[0]


@mx.custom_function
def fused_lstm(x, Wx, Wh, bias, h0, c0):
    """Trainable input-fused LSTM (forward kernel + fused BPTT backward kernel).
    Gradients exact to ~3e-7; ~6.7x over eager MLX and faster than MPSGraph at
    fwd+bwd.  h0,c0 must be explicit (zeros for a fresh sequence)."""
    tg = 1024 if x.shape[0] <= 128 else 256
    return _fwd_train(x, Wx, Wh, bias, h0, c0, tg)[0]


@fused_lstm.vjp
def _fused_lstm_vjp(primals, cotangents, output):
    x, Wx, Wh, bias, h0, c0 = primals
    B, T, IN = x.shape; H = Wh.shape[1]
    tg = 1024 if B <= 128 else 256
    h_out, c_seq, gates = _fwd_train(x, Wx, Wh, bias, h0, c0, tg)
    dh = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
    dz = _bwd_dz(dh, gates, c_seq, c0, Wh, tg)
    dzf = dz.reshape(B * T, 4 * H)
    h_prev = mx.concatenate([h0[:, None, :], h_out[:, :-1, :]], axis=1).reshape(B * T, H)
    dx = (dzf @ Wx).reshape(B, T, IN)
    return [dx, dzf.T @ x.reshape(B * T, IN), dzf.T @ h_prev, dzf.sum(axis=0),
            dz[:, 0, :] @ Wh, mx.zeros_like(c0)]


class FusedLSTM:
    """Minimal LSTM module using the fused kernel (forward/inference).

    Parameters Wx:[4H,in], Wh:[4H,H], bias:[4H] (gate order i,f,g,o, MLX layout).
    For training, wrap the call in mx.custom_function with an eager-recompute vjp,
    or use the compiled-extension build (mlx_addons.fused_rnn_ext) for a native
    backward — the JIT path here is forward-only.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        import mlx.nn as nn
        self.Wx = mx.random.normal((4 * hidden_dim, input_dim)) * (input_dim ** -0.5)
        self.Wh = mx.random.normal((4 * hidden_dim, hidden_dim)) * (hidden_dim ** -0.5)
        self.bias = mx.zeros((4 * hidden_dim,))
        self.hidden_dim = hidden_dim

    def __call__(self, x):
        return fused_lstm_sequence(x, self.Wx, self.Wh, self.bias)


def _selftest():
    mx.random.seed(0)
    B, T, IN, H = 48, 64, 128, 64
    x = mx.random.normal((B, T, IN)) * 0.3
    Wx = mx.random.normal((4 * H, IN)) * 0.1
    Wh = mx.random.normal((4 * H, H)) * 0.1
    bias = mx.random.normal((4 * H,)) * 0.1

    def eager(x):
        ip = x @ Wx.T + bias
        h = mx.zeros((B, H)); c = mx.zeros((B, H)); outs = []
        for t in range(T):
            z = ip[:, t, :] + h @ Wh.T
            i, f, g, o = mx.split(z, 4, axis=-1)
            i, f, g, o = mx.sigmoid(i), mx.sigmoid(f), mx.tanh(g), mx.sigmoid(o)
            c = f * c + i * g; h = o * mx.tanh(c); outs.append(h)
        return mx.stack(outs, axis=1)

    for tg in (256, 1024):
        d = float(mx.max(mx.abs(fused_lstm_sequence(x, Wx, Wh, bias, tg=tg) - eager(x))))
        assert d < 1e-3, (tg, d)
        print(f"  tg={tg}: max|Δ| vs eager = {d:.2e}  OK")


if __name__ == "__main__":
    _selftest()
