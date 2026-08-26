// Copyright © 2026
#pragma once
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;
namespace mlx_fused_lstm {

// ---- ops ----
// Input-projection-fused LSTM. Returns {h[B,T,H], c[B,T,H], gates[B,T,4H]}.
std::vector<mx::array> fused_lstm_fwd(
    const mx::array& x, const mx::array& Wx, const mx::array& Wh,
    const mx::array& bias, const mx::array& h0, const mx::array& c0,
    mx::StreamOrDevice s = {});

// Fused BPTT backward. Returns dz[B,T,4H] (gate pre-activation grads).
mx::array fused_lstm_bwd(
    const mx::array& dh_seq, const mx::array& gates, const mx::array& c_seq,
    const mx::array& c0, const mx::array& Wh, mx::StreamOrDevice s = {});

// ---- primitives (plain mx::Primitive: the exported base, linkable from an
// extension; eval_gpu launches the same Metal kernel as fast::Custom would) ----
class FusedLSTMFwd : public mx::Primitive {
 public:
  explicit FusedLSTMFwd(mx::Stream stream) : mx::Primitive(stream) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("FusedLSTMFwd: GPU only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  std::vector<mx::array> jvp(const std::vector<mx::array>&,
                             const std::vector<mx::array>&,
                             const std::vector<int>&) override {
    throw std::runtime_error("FusedLSTMFwd: jvp NYI");
  }
  std::vector<mx::array> vjp(
      const std::vector<mx::array>& primals,
      const std::vector<mx::array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<mx::array>& outputs) override;
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>&, const std::vector<int>&) override {
    throw std::runtime_error("FusedLSTMFwd: vmap NYI");
  }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    int B = inputs[0].shape(0), T = inputs[0].shape(1), H = inputs[4].shape(1);
    return {{B, T, H}, {B, T, H}, {B, T, 4 * H}};
  }
  const char* name() const override { return "FusedLSTMFwd"; }
  bool is_equivalent(const mx::Primitive& o) const override {
    return o.name() == name();
  }
};

class FusedLSTMBwd : public mx::Primitive {
 public:
  explicit FusedLSTMBwd(mx::Stream stream) : mx::Primitive(stream) {}
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("FusedLSTMBwd: GPU only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  std::vector<mx::array> jvp(const std::vector<mx::array>&,
                             const std::vector<mx::array>&,
                             const std::vector<int>&) override {
    throw std::runtime_error("FusedLSTMBwd: jvp NYI");
  }
  std::vector<mx::array> vjp(const std::vector<mx::array>&,
                             const std::vector<mx::array>&,
                             const std::vector<int>&,
                             const std::vector<mx::array>&) override {
    throw std::runtime_error("FusedLSTMBwd: vjp NYI");
  }
  std::pair<std::vector<mx::array>, std::vector<int>> vmap(
      const std::vector<mx::array>&, const std::vector<int>&) override {
    throw std::runtime_error("FusedLSTMBwd: vmap NYI");
  }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    int B = inputs[0].shape(0), T = inputs[0].shape(1), H = inputs[0].shape(2);
    return {{B, T, 4 * H}};
  }
  const char* name() const override { return "FusedLSTMBwd"; }
  bool is_equivalent(const mx::Primitive& o) const override {
    return o.name() == name();
  }
};

} // namespace mlx_fused_lstm
