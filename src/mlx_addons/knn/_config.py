"""Configuration dataclasses for KNN."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegularizationConfig:
    """Controls regularization of tree nodes by maximum volume."""
    regularize_percentile: float = 90.0
    max_volume_fac: float = 20.0


@dataclass(frozen=True)
class TreeConfig:
    """Controls tree structure and allocation."""
    max_leaf_size: int = 32
    coarse_fac: float = 6.0
    alloc_fac_nodes: float = 1.0
    regularization: RegularizationConfig | None = None


@dataclass(frozen=True)
class FofCatalogueConfig:
    """Controls Friends-of-Friends catalogue properties."""
    npart_min: int = 20


@dataclass(frozen=True)
class FofConfig:
    """Main config for Friends-of-Friends algorithm."""
    alloc_fac_ilist: float = 32.0

    tree: TreeConfig = TreeConfig(
        max_leaf_size=48,
        coarse_fac=8.0,
        alloc_fac_nodes=1.1,
    )
    catalogue: FofCatalogueConfig = FofCatalogueConfig()


@dataclass(frozen=True)
class KNNConfig:
    """Main config for K-Nearest Neighbor search."""
    alloc_fac_ilist: float = 256.0

    tree: TreeConfig = TreeConfig(
        max_leaf_size=48,
        coarse_fac=8.0,
        alloc_fac_nodes=1.0,
        regularization=RegularizationConfig(),
    )
