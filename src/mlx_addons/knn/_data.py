"""Data structures for KNN.

Simplified single-GPU versions of jztree's data classes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx

from ._tools import cumsum_starting_with_zero, set_range, inverse_of_splits


# ---------------------------------------------------------------------------
#  Particle data classes
# ---------------------------------------------------------------------------

@dataclass
class Pos:
    """Holds positions (N, dim)."""
    pos: mx.array


@dataclass
class PosMass:
    """Holds positions and masses."""
    pos: mx.array
    mass: mx.array


@dataclass
class ParticleData:
    """Holds positions and optional per-particle data."""
    pos: mx.array
    mass: Optional[mx.array] = None
    vel: Optional[mx.array] = None
    id: Optional[mx.array] = None


# ---------------------------------------------------------------------------
#  Helpers for accessing particle positions
# ---------------------------------------------------------------------------

def get_pos(part) -> mx.array:
    """Extract positions from any particle-like object."""
    if isinstance(part, mx.array):
        assert part.ndim == 2 and part.shape[-1] <= 10
        return part
    elif hasattr(part, "pos"):
        assert part.pos.ndim == 2
        return part.pos
    else:
        raise ValueError("Invalid particle input — need .pos or an (N,dim) array")


def get_num(part) -> int:
    """Number of particles (just the array length for single-GPU)."""
    return get_pos(part).shape[0]


# ---------------------------------------------------------------------------
#  Morton level info
# ---------------------------------------------------------------------------

def _min_max_msb_diff(dtype):
    if dtype == mx.float32:
        return -150, 128
    elif dtype == mx.float64:
        return -1075, 1024
    elif dtype == mx.int32:
        return -1, 32
    elif dtype == mx.int64:
        return -1, 64
    else:
        return -150, 128  # default to float32


@dataclass(frozen=True)
class LevelInfo:
    """Min/max Morton tree level for given dimensionality and dtype."""
    dim: int
    dtype: mx.Dtype

    def min_lvl(self) -> int:
        return self.dim * _min_max_msb_diff(self.dtype)[0]

    def max_lvl(self) -> int:
        return self.dim * (_min_max_msb_diff(self.dtype)[1] + 1)


# ---------------------------------------------------------------------------
#  Packed array — stacks multiple levels in one contiguous buffer
# ---------------------------------------------------------------------------

@dataclass
class PackedArray:
    """Stacks arrays from multiple tree levels into one contiguous buffer.

    Level *i* occupies data[ispl[i] : ispl[i+1]].
    """
    data: mx.array
    ispl: mx.array  # (nlevels+1,) int32 — level boundaries
    fill_values: Optional[mx.array] = None  # per-level fill value

    # --- constructors -------------------------------------------------------

    @classmethod
    def create_empty(cls, size, levels: int, dtype=mx.float32, fill_values=None):
        data = mx.zeros(size if isinstance(size, int) else size, dtype=dtype)
        ispl = mx.zeros(levels + 1, dtype=mx.int32)
        if fill_values is not None and not isinstance(fill_values, mx.array):
            fill_values = mx.array([fill_values] * levels, dtype=dtype)
        return cls(data, ispl, fill_values)

    @classmethod
    def from_data(cls, data, ispl, fill_values=None):
        if fill_values is not None and not isinstance(fill_values, mx.array):
            if hasattr(fill_values, '__len__'):
                fill_values = mx.array(fill_values, dtype=data.dtype)
            else:
                nlevels = len(ispl) - 1
                fill_values = mx.full(nlevels, fill_values, dtype=data.dtype)
        return cls(data, ispl, fill_values)

    # --- accessors ----------------------------------------------------------

    def get(self, level: int, size: Optional[int] = None, fill_value=None):
        """Get elements for *level*, padded to *size*."""
        if size is None:
            size = self.size()
        indices = mx.arange(size, dtype=mx.int32) + self.ispl[level]
        valid = indices < self.ispl[level + 1]
        valid = valid.reshape((-1,) + (1,) * (self.data.ndim - 1))

        if fill_value is None and self.fill_values is not None:
            fill_value = self.fill_values[level].astype(self.data.dtype)
        elif fill_value is None:
            fill_value = 0

        safe_indices = mx.minimum(indices, len(self.data) - 1)
        return mx.where(valid, self.data[safe_indices], fill_value)

    def set(self, level, values, num=None, fill_value=None):
        """Write *values* into the buffer for *level*."""
        if num is None:
            num = values.shape[0]
        new_ispl = mx.where(
            mx.arange(len(self.ispl)) <= level,
            self.ispl,
            self.ispl[level] + num,
        )
        new_data = set_range(self.data, values, self.ispl[level], self.ispl[level] + num)
        new_fv = self.fill_values
        if fill_value is not None and new_fv is not None:
            new_fv = new_fv.at[level].add(fill_value - new_fv[level])
        return PackedArray(new_data, new_ispl, new_fv)

    def append(self, values, num=None, fill_value=None, resize=False):
        """Append a new level."""
        if resize:
            arr = self.resize_levels(self.nlevels() + 1)
        else:
            arr = self
        nlev = arr.nlevels() - 1  # current last filled
        # find first unfilled level
        return arr.set(nlev, values, num, fill_value)

    def resize_levels(self, levels):
        """Grow the ispl array to accommodate more levels."""
        new_ispl = mx.full(levels + 1, int(self.ispl[-1]), dtype=mx.int32)
        new_ispl = mx.concatenate([self.ispl, new_ispl[len(self.ispl):]])
        fv = self.fill_values
        if fv is not None:
            pad_n = levels - len(fv)
            if pad_n > 0:
                fv = mx.concatenate([fv, mx.zeros(pad_n, dtype=fv.dtype)])
        return PackedArray(self.data, new_ispl, fv)

    # --- queries ------------------------------------------------------------

    def size(self) -> int:
        return self.data.shape[0]

    def num(self, level) -> mx.array:
        return self.ispl[level + 1] - self.ispl[level]

    def nfilled(self) -> mx.array:
        return self.ispl[-1]

    def nlevels(self) -> int:
        return len(self.ispl) - 1


# ---------------------------------------------------------------------------
#  Tree hierarchy
# ---------------------------------------------------------------------------

@dataclass
class TreeHierarchy:
    """Multi-level Z-order tree hierarchy.

    Level 0 = leaves (groups of particles).
    Higher levels = coarser nodes.

    The numpy ``_np`` fields hold the tree structure in numpy for fast
    serial tree-walk logic.  The MLX fields wrap the same data for GPU use.
    """
    size_leaves: int

    ispl_n2n: PackedArray
    ispl_n2l: PackedArray
    ispl_l2p: mx.array

    lvl: PackedArray
    geom_cent: PackedArray

    # --- numpy mirrors (used by the tree-walk in knn/fof) ---
    leaf_splits_np: object = None    # (nleaves+1,) int32
    level_splits_np: object = None   # list of np arrays per level
    n2l_np: object = None            # list of np arrays per level
    bbox_min_np: object = None       # list of (nnodes, dim) np arrays per level
    bbox_max_np: object = None       # list of (nnodes, dim) np arrays per level
    geom_cent_np: object = None      # list of (nnodes, dim) np arrays per level
    level_npart_np: object = None    # list of (nnodes,) np arrays per level

    def splits_leaf_to_part(self, size: Optional[int] = None) -> mx.array:
        if size is None:
            size = self.size() + 1
        return self.ispl_l2p[:size]

    def npart(self, level: int, size: Optional[int] = None) -> mx.array:
        if size is None:
            size = self.size()
        ispl_n2p = self.ispl_l2p[self.ispl_n2l.get(level, size + 1).astype(mx.int32)]
        return ispl_n2p[1:] - ispl_n2p[:-1]

    def num_planes(self) -> int:
        return self.ispl_n2l.nlevels()

    def num(self, level) -> mx.array:
        return self.lvl.num(level)

    def size(self) -> int:
        return self.ispl_n2n.size() - 1

    def info(self) -> LevelInfo:
        dim = self.geom_cent.data.shape[-1]
        dtype = self.geom_cent.data.dtype
        return LevelInfo(dim, dtype)


# ---------------------------------------------------------------------------
#  Interaction list
# ---------------------------------------------------------------------------

@dataclass
class InteractionList:
    """Interaction information for dual tree-walks.

    Node *i* interacts with source nodes isrc[ispl[i]:ispl[i+1]].
    """
    ispl: mx.array   # (nnodes+1,) int32
    isrc: mx.array   # interaction source indices

    rad2: Optional[mx.array] = None  # squared radii (KNN)

    def nfilled(self) -> mx.array:
        return self.ispl[-1]

    def size(self) -> int:
        return self.isrc.shape[0]


# ---------------------------------------------------------------------------
#  Node data helpers
# ---------------------------------------------------------------------------

@dataclass
class PosLvl:
    pos: mx.array
    lvl: mx.array


@dataclass
class PosLvlNum:
    pos: mx.array
    lvl: mx.array
    npart: mx.array


@dataclass
class FofNodeData:
    lvl: mx.array
    igroup: mx.array
    spl: mx.array


# ---------------------------------------------------------------------------
#  FoF catalogue
# ---------------------------------------------------------------------------

@dataclass
class FofCatalogue:
    """Friends-of-Friends group catalogue."""
    ngroups: mx.array
    mass: Optional[mx.array] = None
    count: Optional[mx.array] = None
    offset: Optional[mx.array] = None
    com_pos: Optional[mx.array] = None
    com_vel: Optional[mx.array] = None
    com_inertia_radius: Optional[mx.array] = None
