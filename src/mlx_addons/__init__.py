"""
mlx-addons: GPU-accelerated operations for MLX on Apple Silicon.

Modules:
    linalg    - Batched linear algebra via Metal GPU kernels (solve, cholesky)
    knn       - K-nearest neighbors via Z-order tree + Metal GPU kernels
    nndescent - Approximate k-NN graph construction via NNDescent (pure MLX)
    recurrent - Fast LSTM (single + grouped) via fused Metal cell kernels
    fused_rnn - Input-projection-fused whole-sequence LSTM (one JIT Metal kernel;
                forward + trainable fused-BPTT backward). Matches/beats MPSGraph.
    fused_gru - Input-projection-fused whole-sequence GRU (forward + trainable
                fused-BPTT backward); ~4-7x over eager inference, ~17x training.
"""

__version__ = "0.1.0"

from . import linalg
from . import knn
from . import nndescent
from . import recurrent
from . import fused_rnn
from .fused_rnn import fused_lstm, fused_lstm_sequence
from .fused_gru import fused_gru, fused_gru_sequence  # noqa: F401 (fn shadows submodule)
