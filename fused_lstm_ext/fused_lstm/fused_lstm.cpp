// Copyright © 2026
#include <dlfcn.h>
#include <filesystem>

#include "mlx/backend/metal/device.h"
#include "mlx/utils.h"

#include "fused_lstm/fused_lstm.h"

namespace mlx_fused_lstm {

namespace {
std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info))
      throw std::runtime_error("Unable to get current binary dir.");
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}
} // namespace

// ----------------------------- ops -----------------------------
std::vector<mx::array> fused_lstm_fwd(
    const mx::array& x, const mx::array& Wx, const mx::array& Wh,
    const mx::array& bias, const mx::array& h0, const mx::array& c0,
    mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  int B = x.shape(0), T = x.shape(1), H = h0.shape(1), H4 = 4 * H;
  auto xc = mx::contiguous(x, false, s);
  auto Wx_t = mx::contiguous(mx::transpose(Wx, s), false, s); // [IN,4H]
  auto Wh_t = mx::contiguous(mx::transpose(Wh, s), false, s); // [H,4H]
  auto bc = mx::contiguous(bias, false, s);
  auto h0c = mx::contiguous(h0, false, s);
  auto c0c = mx::contiguous(c0, false, s);
  return mx::array::make_arrays(
      {{B, T, H}, {B, T, H}, {B, T, H4}},
      {mx::float32, mx::float32, mx::float32},
      std::make_shared<FusedLSTMFwd>(s),
      {xc, Wx_t, Wh_t, bc, h0c, c0c});
}

mx::array fused_lstm_bwd(
    const mx::array& dh_seq, const mx::array& gates, const mx::array& c_seq,
    const mx::array& c0, const mx::array& Wh, mx::StreamOrDevice s_) {
  auto s = mx::to_stream(s_);
  int B = dh_seq.shape(0), T = dh_seq.shape(1), H = dh_seq.shape(2);
  return mx::array(
      {B, T, 4 * H}, mx::float32,
      std::make_shared<FusedLSTMBwd>(s),
      {mx::contiguous(dh_seq, false, s), mx::contiguous(gates, false, s),
       mx::contiguous(c_seq, false, s), mx::contiguous(c0, false, s),
       mx::contiguous(Wh, false, s)});
}

// ----------------------------- eval_gpu -----------------------------
void FusedLSTMFwd::eval_gpu(
    const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  const auto& x = inputs[0];
  int B = x.shape(0), T = x.shape(1), IN = x.shape(2), H = inputs[4].shape(1), H4 = 4 * H;
  auto& out_h = outputs[0];
  auto& out_c = outputs[1];
  auto& out_g = outputs[2];
  out_h.set_data(mx::allocator::malloc(out_h.nbytes()));
  out_c.set_data(mx::allocator::malloc(out_c.nbytes()));
  out_g.set_data(mx::allocator::malloc(out_g.nbytes()));

  auto lib = d.get_library("fused_lstm_ext", current_binary_dir());
  auto kernel = d.get_kernel("fused_lstm_fwd_f32", lib);
  uint32_t b_tile = 8, b_tile_pad = 8;
  uint32_t num_tgs = (static_cast<uint32_t>(B) + 7) / 8;
  uint32_t tg = (B <= 128) ? 1024u : 256u;
  tg = std::min(tg, static_cast<uint32_t>(kernel->maxTotalThreadsPerThreadgroup()));
  uint32_t Bu = B, Tu = T, Hu = H, INu = IN;

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 6; ++i) enc.set_input_array(inputs[i], i);
  enc.set_output_array(out_h, 6);
  enc.set_output_array(out_c, 7);
  enc.set_output_array(out_g, 8);
  enc.set_bytes(Bu, 9); enc.set_bytes(Tu, 10); enc.set_bytes(Hu, 11); enc.set_bytes(INu, 12);
  enc.set_bytes(b_tile, 13); enc.set_bytes(b_tile_pad, 14);
  enc.set_threadgroup_memory_length(b_tile_pad * Hu * sizeof(float), 0);
  enc.set_threadgroup_memory_length(b_tile_pad * INu * sizeof(float), 1);
  enc.set_threadgroup_memory_length(b_tile_pad * H4 * sizeof(float), 2);
  enc.dispatch_threadgroups(MTL::Size(num_tgs, 1, 1), MTL::Size(tg, 1, 1));
}

void FusedLSTMBwd::eval_gpu(
    const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) {
  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  const auto& dh = inputs[0];
  int B = dh.shape(0), T = dh.shape(1), H = dh.shape(2);
  auto& out_dz = outputs[0];
  out_dz.set_data(mx::allocator::malloc(out_dz.nbytes()));

  auto lib = d.get_library("fused_lstm_ext", current_binary_dir());
  auto kernel = d.get_kernel("fused_lstm_bwd_f32", lib);
  uint32_t b_tile = 8, b_tile_pad = 8;
  uint32_t num_tgs = (static_cast<uint32_t>(B) + 7) / 8;
  uint32_t tg = (B <= 128) ? 1024u : 256u;
  tg = std::min(tg, static_cast<uint32_t>(kernel->maxTotalThreadsPerThreadgroup()));
  uint32_t Bu = B, Tu = T, Hu = H, H4 = 4 * H;

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  for (int i = 0; i < 5; ++i) enc.set_input_array(inputs[i], i);
  enc.set_output_array(out_dz, 5);
  enc.set_bytes(Bu, 6); enc.set_bytes(Tu, 7); enc.set_bytes(Hu, 8);
  enc.set_bytes(b_tile, 9); enc.set_bytes(b_tile_pad, 10);
  enc.set_threadgroup_memory_length(b_tile_pad * H4 * sizeof(float), 0);
  enc.set_threadgroup_memory_length(b_tile_pad * Hu * sizeof(float), 1);
  enc.set_threadgroup_memory_length(b_tile_pad * Hu * sizeof(float), 2);
  enc.dispatch_threadgroups(MTL::Size(num_tgs, 1, 1), MTL::Size(tg, 1, 1));
}

// ----------------------------- vjp (no recompute) -----------------------------
std::vector<mx::array> FusedLSTMFwd::vjp(
    const std::vector<mx::array>& primals,
    const std::vector<mx::array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<mx::array>& outputs) {
  auto s = stream();
  const auto& x = primals[0];
  const auto& Wx_t = primals[1];
  const auto& Wh_t = primals[2];
  const auto& h0 = primals[4];
  const auto& c0 = primals[5];
  int B = x.shape(0), T = x.shape(1), IN = x.shape(2), H = h0.shape(1), H4 = 4 * H;
  const auto& h_out = outputs[0];
  const auto& c_seq = outputs[1];
  const auto& gates = outputs[2];
  auto dh = cotangents[0]; // cotangent of h (c,g cotangents unused)

  auto Wh = mx::transpose(Wh_t, s);                          // [4H,H]
  auto dz = fused_lstm_bwd(dh, gates, c_seq, c0, Wh, s);     // [B,T,4H]
  auto dzf = mx::reshape(dz, {B * T, H4}, s);
  auto xf = mx::reshape(x, {B * T, IN}, s);
  auto h0e = mx::reshape(h0, {B, 1, H}, s);
  auto hpre = mx::slice(h_out, {0, 0, 0}, {B, T - 1, H}, s);
  auto h_prev = mx::reshape(mx::concatenate({h0e, hpre}, 1, s), {B * T, H}, s);

  std::vector<mx::array> all = {
      mx::reshape(mx::matmul(dzf, mx::transpose(Wx_t, s), s), {B, T, IN}, s),  // dx
      mx::matmul(mx::transpose(xf, s), dzf, s),                                 // dWx_t
      mx::matmul(mx::transpose(h_prev, s), dzf, s),                             // dWh_t
      mx::sum(dzf, 0, false, s),                                                // dbias
      mx::matmul(mx::reshape(mx::slice(dz, {0, 0, 0}, {B, 1, H4}, s), {B, H4}, s),
                 mx::transpose(Wh_t, s), s),                                    // dh0
      mx::zeros_like(c0, s)};                                                   // dc0
  std::vector<mx::array> out;
  for (int a : argnums) out.push_back(all[a]);
  return out;
}

} // namespace mlx_fused_lstm
