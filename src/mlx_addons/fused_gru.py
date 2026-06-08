"""
fused_gru — input-projection-fused GRU for MLX (Apple Silicon).

Same idea as ``fused_rnn`` (LSTM): one JIT ``mx.fast.metal_kernel`` runs the whole
sequence with the input projection ``x @ Wx`` FUSED into the recurrent loop (no
``[B,T,3H]`` round-trip).  PyTorch GRU convention, gate order **r, z, n**::

    r = sigmoid(x@Wir + b_ir + h@Whr + b_hr)            (reset)
    z = sigmoid(x@Wiz + b_iz + h@Whz + b_hz)            (update)
    n = tanh   (x@Win + b_in + r * (h@Whn + b_hn))      (new)
    h'= (1 - z) * n + z * h

Unlike the LSTM, the reset gate multiplies ONLY the hidden projection of the n
gate, so the input projection ``ip = x@Wx`` and hidden projection ``hp = h@Wh``
must be kept SEPARATE (two simdgroup-matrix GEMM tiles, not summed).

    Wx:[3H,IN]  Wh:[3H,H]  bias_ih:[3H]  bias_hh:[3H]  h0:[B,H]
    fused_gru_sequence(...)  -> forward (inference)
    fused_gru(...)           -> trainable (custom_function + fused BPTT backward)

Requires IN, H, 3H multiples of 8, float32, GPU.
"""
from __future__ import annotations
import mlx.core as mx

__all__ = ["fused_gru_sequence", "fused_gru"]

_HEADER = "#include <metal_simdgroup_matrix>\n#include <metal_math>\n"
_GF, _GFT, _GBW = {}, {}, {}


# --------------------------------------------------------------------------- fwd
def _gru_fwd_src(B, T, H, IN, TG, save):
    H3, NSG = 3 * H, TG // 32
    save_g = (
        "uint o4=bg*Tn*4u*H+t*4u*H+h;"
        "out_g[o4]=r;out_g[o4+H]=z;out_g[o4+2u*H]=ng;out_g[o4+3u*H]=hn;"
        if save else ""
    )
    return f"""
    const uint H={H}u,H3={H3}u,IN={IN}u,Tn={T}u,BSZ={B}u,b_tile=8u,b_tile_pad=8u,TGSZ={TG}u,NSG={NSG}u;
    const uint K_H=H/8u,K_X=IN/8u,N_T=H3/8u,TOT=(b_tile_pad/8u)*N_T;
    uint tid=thread_index_in_threadgroup,sg=tid/32u,tg_idx=threadgroup_position_in_grid.x,b_base=tg_idx*b_tile;
    threadgroup float sh_h[b_tile_pad*{H}u],sh_x[b_tile_pad*{IN}u],ipb[b_tile_pad*{H3}u],hpb[b_tile_pad*{H3}u];
    for(uint i=tid;i<b_tile_pad*H;i+=TGSZ){{uint b=i/H,h=i%H,bg=b_base+b;sh_h[i]=(b<b_tile&&bg<BSZ)?h_init[bg*H+h]:0.0f;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint t=0u;t<Tn;++t){{
      for(uint i=tid;i<b_tile_pad*IN;i+=TGSZ){{uint b=i/IN,k=i%IN,bg=b_base+b;sh_x[i]=(b<b_tile&&bg<BSZ)?x[bg*Tn*IN+t*IN+k]:0.0f;}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint mn=sg;mn<TOT;mn+=NSG){{uint m=mn/N_T,n=mn%N_T;
        simdgroup_matrix<float,8,8> Ch=simdgroup_matrix<float,8,8>(0),Ci=simdgroup_matrix<float,8,8>(0),A,Bm;
        for(uint k=0u;k<K_H;++k){{simdgroup_load(A,sh_h+m*8u*H+k*8u,H);simdgroup_load(Bm,Wh_t+k*8u*H3+n*8u,H3);simdgroup_multiply_accumulate(Ch,A,Bm,Ch);}}
        simdgroup_store(Ch,hpb+m*8u*H3+n*8u,H3);
        for(uint k=0u;k<K_X;++k){{simdgroup_load(A,sh_x+m*8u*IN+k*8u,IN);simdgroup_load(Bm,Wx_t+k*8u*H3+n*8u,H3);simdgroup_multiply_accumulate(Ci,A,Bm,Ci);}}
        simdgroup_store(Ci,ipb+m*8u*H3+n*8u,H3);}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint i=tid;i<b_tile*H;i+=TGSZ){{uint b=i/H,h=i%H,bg=b_base+b;if(bg>=BSZ)continue;
        uint o3=b*H3+h;
        float r=1.0f/(1.0f+metal::exp(-(ipb[o3]+bih[h]+hpb[o3]+bhh[h])));
        float z=1.0f/(1.0f+metal::exp(-(ipb[o3+H]+bih[H+h]+hpb[o3+H]+bhh[H+h])));
        float hn=hpb[o3+2u*H]+bhh[2u*H+h];
        float ng=metal::precise::tanh(ipb[o3+2u*H]+bih[2u*H+h]+r*hn);
        float hp=sh_h[b*H+h];
        float hnew=(1.0f-z)*ng+z*hp;
        out_h[bg*Tn*H+t*H+h]=hnew; sh_h[b*H+h]=hnew;
        {save_g}}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }}"""


# --------------------------------------------------------------------------- bwd
def _gru_bwd_src(B, T, H, TG):
    H3, NSG = 3 * H, TG // 32
    return f"""
    const uint H={H}u,H3={H3}u,Tn={T}u,BSZ={B}u,b_tile=8u,b_tile_pad=8u,TGSZ={TG}u,NSG={NSG}u;
    const uint K_P=H3/8u,N_T=H/8u,TOT=(b_tile_pad/8u)*N_T;
    uint tid=thread_index_in_threadgroup,sg=tid/32u,tg_idx=threadgroup_position_in_grid.x,b_base=tg_idx*b_tile;
    threadgroup float sh_dhp[b_tile_pad*{H3}u],sh_dhn[b_tile_pad*{H}u],sh_dir[b_tile_pad*{H}u];
    for(uint i=tid;i<b_tile_pad*H;i+=TGSZ){{sh_dhn[i]=0.0f;sh_dir[i]=0.0f;}}
    for(uint i=tid;i<b_tile_pad*H3;i+=TGSZ){{sh_dhp[i]=0.0f;}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int tt=int(Tn)-1;tt>=0;--tt){{uint t=uint(tt);
      for(uint i=tid;i<b_tile*H;i+=TGSZ){{uint b=i/H,h=i%H,bg=b_base+b;if(bg>=BSZ)continue;
        uint g4=bg*Tn*4u*H+t*4u*H+h;
        float r=gates[g4],z=gates[g4+H],ng=gates[g4+2u*H],hn=gates[g4+3u*H];
        float hp=(t==0u)?h_init[bg*H+h]:out_h_seq[bg*Tn*H+(t-1u)*H+h];
        float dh=dh_seq[bg*Tn*H+t*H+h]+sh_dhn[b*H+h];
        float dn=dh*(1.0f-z);
        float dzg=dh*(hp-ng);
        float dpre_n=dn*(1.0f-ng*ng);
        float dxn=dpre_n,dhn=dpre_n*r,dr=dpre_n*hn;
        float dpre_r=dr*r*(1.0f-r);
        float dpre_z=dzg*z*(1.0f-z);
        uint o3=b*H3+h,gp=bg*Tn*H3+t*H3+h;
        out_dip[gp]=dpre_r;out_dip[gp+H]=dpre_z;out_dip[gp+2u*H]=dxn;
        sh_dhp[o3]=dpre_r;sh_dhp[o3+H]=dpre_z;sh_dhp[o3+2u*H]=dhn;
        out_dhp[gp]=dpre_r;out_dhp[gp+H]=dpre_z;out_dhp[gp+2u*H]=dhn;
        sh_dir[b*H+h]=dh*z;}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint mn=sg;mn<TOT;mn+=NSG){{uint m=mn/N_T,n=mn%N_T;simdgroup_matrix<float,8,8> C=simdgroup_matrix<float,8,8>(0),A,Bm;
        for(uint k=0u;k<K_P;++k){{simdgroup_load(A,sh_dhp+m*8u*H3+k*8u,H3);simdgroup_load(Bm,Wh+k*8u*H+n*8u,H);simdgroup_multiply_accumulate(C,A,Bm,C);}}
        simdgroup_store(C,sh_dhn+m*8u*H+n*8u,H);}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint i=tid;i<b_tile_pad*H;i+=TGSZ){{sh_dhn[i]+=sh_dir[i];}}
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if(t==0u){{for(uint i=tid;i<b_tile*H;i+=TGSZ){{uint b=i/H,h=i%H,bg=b_base+b;if(bg<BSZ)out_dh0[bg*H+h]=sh_dhn[b*H+h];}}}}
    }}"""


# --------------------------------------------------------------------------- py
def _defaults(x, Wh, bih, bhh, h0, tg):
    B, T, IN = x.shape
    H = Wh.shape[1]
    if tg == 0:
        tg = 1024 if B <= 128 else 256
    if bih is None:
        bih = mx.zeros((3 * H,), dtype=x.dtype)
    if bhh is None:
        bhh = mx.zeros((3 * H,), dtype=x.dtype)
    if h0 is None:
        h0 = mx.zeros((B, H), dtype=x.dtype)
    return B, T, IN, H, tg, bih, bhh, h0


def fused_gru_sequence(x, Wx, Wh, bias_ih=None, bias_hh=None, h0=None, tg: int = 0):
    """Forward GRU over a full sequence, input projection fused in.

    x:[B,T,IN]  Wx:[3H,IN]  Wh:[3H,H]  bias_ih,bias_hh:[3H] (default 0)  h0:[B,H].
    Returns h_out:[B,T,H].  Requires IN,H,3H multiples of 8, float32, GPU.
    """
    B, T, IN, H, tg, bih, bhh, h0 = _defaults(x, Wh, bias_ih, bias_hh, h0, tg)
    key = (B, T, H, IN, tg)
    if key not in _GF:
        _GF[key] = mx.fast.metal_kernel(
            name=f"fused_gru_xin_{B}_{T}_{H}_{IN}_{tg}",
            input_names=["x", "Wx_t", "Wh_t", "bih", "bhh", "h_init"],
            output_names=["out_h"], header=_HEADER,
            source=_gru_fwd_src(B, T, H, IN, tg, save=False))
    return _GF[key](
        inputs=[mx.contiguous(x), mx.contiguous(Wx.T), mx.contiguous(Wh.T), bih, bhh, h0],
        output_shapes=[(B, T, H)], output_dtypes=[x.dtype],
        grid=((B + 7) // 8 * tg, 1, 1), threadgroup=(tg, 1, 1))[0]


def _gru_fwd_train(x, Wx, Wh, bih, bhh, h0, tg):
    B, T, IN = x.shape
    H = Wh.shape[1]
    key = (B, T, H, IN, tg)
    if key not in _GFT:
        _GFT[key] = mx.fast.metal_kernel(
            name=f"fused_gru_ft_{B}_{T}_{H}_{IN}_{tg}",
            input_names=["x", "Wx_t", "Wh_t", "bih", "bhh", "h_init"],
            output_names=["out_h", "out_g"], header=_HEADER,
            source=_gru_fwd_src(B, T, H, IN, tg, save=True))
    return _GFT[key](
        inputs=[mx.contiguous(x), mx.contiguous(Wx.T), mx.contiguous(Wh.T), bih, bhh, h0],
        output_shapes=[(B, T, H), (B, T, 4 * H)], output_dtypes=[x.dtype, x.dtype],
        grid=((B + 7) // 8 * tg, 1, 1), threadgroup=(tg, 1, 1))


def _gru_bwd(dh_seq, gates, h_out, h0, Wh, tg):
    B, T, H = dh_seq.shape
    key = (B, T, H, tg)
    if key not in _GBW:
        _GBW[key] = mx.fast.metal_kernel(
            name=f"fused_gru_bw_{B}_{T}_{H}_{tg}",
            input_names=["dh_seq", "gates", "out_h_seq", "h_init", "Wh"],
            output_names=["out_dip", "out_dhp", "out_dh0"], header=_HEADER,
            source=_gru_bwd_src(B, T, H, tg))
    return _GBW[key](
        inputs=[mx.contiguous(dh_seq), gates, h_out, h0, mx.contiguous(Wh)],
        output_shapes=[(B, T, 3 * H), (B, T, 3 * H), (B, H)],
        output_dtypes=[dh_seq.dtype] * 3,
        grid=((B + 7) // 8 * tg, 1, 1), threadgroup=(tg, 1, 1))


@mx.custom_function
def fused_gru(x, Wx, Wh, bias_ih, bias_hh, h0):
    """Trainable input-fused GRU (forward kernel + fused BPTT backward kernel).

    x:[B,T,IN] Wx:[3H,IN] Wh:[3H,H] bias_ih:[3H] bias_hh:[3H] h0:[B,H].
    Returns h_out:[B,T,H].  Gradients exact to ~1e-6 vs eager.  PyTorch GRU
    convention (gate order r,z,n; n = tanh(x@Win+b_in + r*(h@Whn+b_hn))).
    """
    tg = 1024 if x.shape[0] <= 128 else 256
    return _gru_fwd_train(x, Wx, Wh, bias_ih, bias_hh, h0, tg)[0]


@fused_gru.vjp
def _gru_vjp(primals, cotangents, output):
    x, Wx, Wh, bih, bhh, h0 = primals
    B, T, IN = x.shape
    H = Wh.shape[1]
    tg = 1024 if B <= 128 else 256
    h_out, gates = _gru_fwd_train(x, Wx, Wh, bih, bhh, h0, tg)
    dh = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
    dip, dhp, dh0 = _gru_bwd(dh, gates, h_out, h0, Wh, tg)
    dipf = dip.reshape(B * T, 3 * H)
    dhpf = dhp.reshape(B * T, 3 * H)
    h_prev = mx.concatenate([h0[:, None, :], h_out[:, :-1, :]], axis=1).reshape(B * T, H)
    dx = (dipf @ Wx).reshape(B, T, IN)
    return [dx, dipf.T @ x.reshape(B * T, IN), dhpf.T @ h_prev,
            dipf.sum(axis=0), dhpf.sum(axis=0), dh0]


def _selftest():
    mx.random.seed(0)
    B, T, IN, H = 48, 32, 128, 64

    def randn(*s, k=1.0):
        return mx.random.normal(s) * k

    x = randn(B, T, IN, k=0.5)
    Wx = randn(3 * H, IN, k=0.1)
    Wh = randn(3 * H, H, k=0.1)
    bih = randn(3 * H, k=0.1)
    bhh = randn(3 * H, k=0.1)
    h0 = randn(B, H, k=0.1)

    def eager(x, Wx, Wh, bih, bhh, h0):
        xall = x @ Wx.T
        h = h0
        outs = []
        for t in range(T):
            xt = xall[:, t, :]
            hp = h @ Wh.T
            r = mx.sigmoid(xt[:, :H] + bih[:H] + hp[:, :H] + bhh[:H])
            z = mx.sigmoid(xt[:, H:2 * H] + bih[H:2 * H] + hp[:, H:2 * H] + bhh[H:2 * H])
            n = mx.tanh(xt[:, 2 * H:] + bih[2 * H:] + r * (hp[:, 2 * H:] + bhh[2 * H:]))
            h = (1 - z) * n + z * h
            outs.append(h)
        return mx.stack(outs, axis=1)

    he = eager(x, Wx, Wh, bih, bhh, h0)
    hf = fused_gru(x, Wx, Wh, bih, bhh, h0)
    mx.eval(he, hf)
    rel = lambda a, b: float(mx.max(mx.abs(a - b)) / (mx.max(mx.abs(b)) + 1e-8))
    print(f"  forward rel_err = {rel(hf, he):.2e}")

    cot = randn(B, T, H)
    ge = mx.grad(lambda *p: mx.sum(eager(*p) * cot), argnums=(0, 1, 2, 3, 4, 5))(x, Wx, Wh, bih, bhh, h0)
    gf = mx.grad(lambda *p: mx.sum(fused_gru(*p) * cot), argnums=(0, 1, 2, 3, 4, 5))(x, Wx, Wh, bih, bhh, h0)
    mx.eval(ge, gf)
    for nm, a, b in zip(["dx", "dWx", "dWh", "dbih", "dbhh", "dh0"], gf, ge):
        print(f"  {nm:5s} rel_err = {rel(a, b):.2e}")


if __name__ == "__main__":
    _selftest()
