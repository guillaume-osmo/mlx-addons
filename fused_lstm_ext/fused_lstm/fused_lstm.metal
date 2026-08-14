// Input-projection-fused LSTM forward (saves gates) + fused BPTT backward.
// fast::Custom-style kernels: runtime dims + dynamic threadgroup memory.
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// gates = x[t]@Wx_t + h@Wh_t + bias ; i,f,g,o ; c=f*c+i*g ; h=o*tanh(c).
// Saves out_g[B,T,4H] = post-activation (i,f,g,o) for the backward.
[[kernel]] void fused_lstm_fwd_f32(
    device const float* x        [[buffer(0)]],    // [B,T,IN]
    device const float* Wx_t     [[buffer(1)]],    // [IN,4H]
    device const float* Wh_t     [[buffer(2)]],    // [H,4H]
    device const float* bias     [[buffer(3)]],    // [4H]
    device const float* h_init   [[buffer(4)]],    // [B,H]
    device const float* c_init   [[buffer(5)]],    // [B,H]
    device float* out_h          [[buffer(6)]],    // [B,T,H]
    device float* out_c          [[buffer(7)]],    // [B,T,H]
    device float* out_g          [[buffer(8)]],    // [B,T,4H]
    constant uint& B             [[buffer(9)]],
    constant uint& T             [[buffer(10)]],
    constant uint& H             [[buffer(11)]],
    constant uint& IN            [[buffer(12)]],
    constant uint& b_tile        [[buffer(13)]],
    constant uint& b_tile_pad    [[buffer(14)]],
    threadgroup float* sh_h      [[threadgroup(0)]],
    threadgroup float* sh_x      [[threadgroup(1)]],
    threadgroup float* gbuf      [[threadgroup(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]],
    uint nsg    [[simdgroups_per_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint TGSZ   [[threads_per_threadgroup]]) {
  uint H4 = 4u * H, K_H = H / 8u, K_X = IN / 8u, N_T = H4 / 8u, TOT = (b_tile_pad / 8u) * N_T;
  uint b_base = tg_idx * b_tile;
  for (uint i = tid; i < b_tile_pad * H; i += TGSZ) {
    uint b = i / H, h = i % H, bg = b_base + b;
    sh_h[i] = (b < b_tile && bg < B) ? h_init[bg * H + h] : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint t = 0u; t < T; ++t) {
    for (uint i = tid; i < b_tile_pad * IN; i += TGSZ) {
      uint b = i / IN, k = i % IN, bg = b_base + b;
      sh_x[i] = (b < b_tile && bg < B) ? x[bg * T * IN + t * IN + k] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint mn = sg; mn < TOT; mn += nsg) {
      uint m = mn / N_T, n = mn % N_T;
      simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0), A, Bm;
      for (uint k = 0u; k < K_H; ++k) {
        simdgroup_load(A, sh_h + m * 8u * H + k * 8u, H);
        simdgroup_load(Bm, Wh_t + k * 8u * H4 + n * 8u, H4);
        simdgroup_multiply_accumulate(C, A, Bm, C);
      }
      for (uint k = 0u; k < K_X; ++k) {
        simdgroup_load(A, sh_x + m * 8u * IN + k * 8u, IN);
        simdgroup_load(Bm, Wx_t + k * 8u * H4 + n * 8u, H4);
        simdgroup_multiply_accumulate(C, A, Bm, C);
      }
      simdgroup_store(C, gbuf + m * 8u * H4 + n * 8u, H4);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < b_tile * H; i += TGSZ) {
      uint b = i / H, h = i % H, bg = b_base + b;
      if (bg >= B) continue;
      float ig = 1.0f / (1.0f + exp(-(gbuf[b * H4 + h] + bias[h])));
      float fg = 1.0f / (1.0f + exp(-(gbuf[b * H4 + H + h] + bias[H + h])));
      float gv = precise::tanh(gbuf[b * H4 + 2u * H + h] + bias[2u * H + h]);
      float og = 1.0f / (1.0f + exp(-(gbuf[b * H4 + 3u * H + h] + bias[3u * H + h])));
      float cp = (t == 0u) ? c_init[bg * H + h] : out_c[bg * T * H + (t - 1u) * H + h];
      float cn = fg * cp + ig * gv, hn = og * precise::tanh(cn);
      uint o = bg * T * H + t * H + h;
      out_h[o] = hn; out_c[o] = cn; sh_h[b * H + h] = hn;
      uint o4 = bg * T * H4 + t * H4 + h;
      out_g[o4] = ig; out_g[o4 + H] = fg; out_g[o4 + 2u * H] = gv; out_g[o4 + 3u * H] = og;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// reverse-time BPTT -> dz[B,T,4H].  dh_next = dz@Wh ; dc_next = dc*f.
[[kernel]] void fused_lstm_bwd_f32(
    device const float* dh_seq   [[buffer(0)]],    // [B,T,H]
    device const float* gates    [[buffer(1)]],    // [B,T,4H]
    device const float* c_seq    [[buffer(2)]],    // [B,T,H]
    device const float* c_init   [[buffer(3)]],    // [B,H]
    device const float* Wh       [[buffer(4)]],    // [4H,H]
    device float* out_dz         [[buffer(5)]],    // [B,T,4H]
    constant uint& B             [[buffer(6)]],
    constant uint& T             [[buffer(7)]],
    constant uint& H             [[buffer(8)]],
    constant uint& b_tile        [[buffer(9)]],
    constant uint& b_tile_pad    [[buffer(10)]],
    threadgroup float* sh_dz     [[threadgroup(0)]],
    threadgroup float* sh_dhn    [[threadgroup(1)]],
    threadgroup float* sh_dcn    [[threadgroup(2)]],
    uint tid    [[thread_index_in_threadgroup]],
    uint sg     [[simdgroup_index_in_threadgroup]],
    uint nsg    [[simdgroups_per_threadgroup]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint TGSZ   [[threads_per_threadgroup]]) {
  uint H4 = 4u * H, K_Z = H4 / 8u, N_T = H / 8u, TOT = (b_tile_pad / 8u) * N_T;
  uint b_base = tg_idx * b_tile;
  for (uint i = tid; i < b_tile_pad * H; i += TGSZ) { sh_dhn[i] = 0.0f; sh_dcn[i] = 0.0f; }
  for (uint i = tid; i < b_tile_pad * H4; i += TGSZ) { sh_dz[i] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int tt = int(T) - 1; tt >= 0; --tt) {
    uint t = uint(tt);
    for (uint i = tid; i < b_tile * H; i += TGSZ) {
      uint b = i / H, h = i % H, bg = b_base + b;
      if (bg >= B) continue;
      uint g4 = bg * T * H4 + t * H4 + h;
      float ig = gates[g4], fg = gates[g4 + H], gv = gates[g4 + 2u * H], og = gates[g4 + 3u * H];
      float c = c_seq[bg * T * H + t * H + h];
      float cp = (t == 0u) ? c_init[bg * H + h] : c_seq[bg * T * H + (t - 1u) * H + h];
      float tc = precise::tanh(c);
      float dh = dh_seq[bg * T * H + t * H + h] + sh_dhn[b * H + h];
      float dgo = dh * tc;
      float dc = dh * og * (1.0f - tc * tc) + sh_dcn[b * H + h];
      float dzi = dc * gv * ig * (1.0f - ig);
      float dzf = dc * cp * fg * (1.0f - fg);
      float dzg = dc * ig * (1.0f - gv * gv);
      float dzo = dgo * og * (1.0f - og);
      uint zb = b * H4 + h;
      sh_dz[zb] = dzi; sh_dz[zb + H] = dzf; sh_dz[zb + 2u * H] = dzg; sh_dz[zb + 3u * H] = dzo;
      out_dz[g4] = dzi; out_dz[g4 + H] = dzf; out_dz[g4 + 2u * H] = dzg; out_dz[g4 + 3u * H] = dzo;
      sh_dcn[b * H + h] = dc * fg;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint mn = sg; mn < TOT; mn += nsg) {
      uint m = mn / N_T, n = mn % N_T;
      simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0), A, Bm;
      for (uint k = 0u; k < K_Z; ++k) {
        simdgroup_load(A, sh_dz + m * 8u * H4 + k * 8u, H4);
        simdgroup_load(Bm, Wh + k * 8u * H + n * 8u, H);
        simdgroup_multiply_accumulate(C, A, Bm, C);
      }
      simdgroup_store(C, sh_dhn + m * 8u * H + n * 8u, H);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
