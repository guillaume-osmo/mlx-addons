#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include "fused_lstm/fused_lstm.h"
namespace nb = nanobind;
using namespace nb::literals;
NB_MODULE(_ext, m) {
  m.doc() = "Input-fused LSTM (forward + fused BPTT backward) for MLX";
  m.def("fused_lstm_fwd", &mlx_fused_lstm::fused_lstm_fwd,
        "x"_a, "Wx"_a, "Wh"_a, "bias"_a, "h0"_a, "c0"_a,
        nb::kw_only(), "stream"_a = nb::none());
  m.def("fused_lstm_bwd", &mlx_fused_lstm::fused_lstm_bwd,
        "dh_seq"_a, "gates"_a, "c_seq"_a, "c0"_a, "Wh"_a,
        nb::kw_only(), "stream"_a = nb::none());
}
