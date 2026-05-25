"""
mlx-addons: GPU-accelerated operations for MLX on Apple Silicon.

Modules:
    linalg     - Batched linear algebra via Metal GPU kernels (solve, cholesky)
    knn        - K-nearest neighbors via Z-order tree + Metal GPU kernels
    nndescent  - Approximate k-NN graph construction via NNDescent (pure MLX)
    recurrent  - Fast LSTM (single + grouped) via fused Metal cell kernels
    similarity - Cosine/Tanimoto similarity + online clustering for novelty filters
"""

__version__ = "0.1.0"

from . import linalg
from . import knn
from . import nndescent
from . import recurrent
from . import similarity
