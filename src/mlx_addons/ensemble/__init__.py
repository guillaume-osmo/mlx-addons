"""Tree ensembles on Apple GPU via a CSR / segment-scatter formulation."""

from ._csr_trees import ExtraTreesRegressorMLXCSR

__all__ = ["ExtraTreesRegressorMLXCSR"]
