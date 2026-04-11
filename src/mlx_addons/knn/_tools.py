"""Utility functions for KNN."""

import mlx.core as mx


def where_true(condition: mx.array) -> mx.array:
    """Return indices where condition is True.

    MLX's mx.where requires 3 args, so we implement the 1-arg numpy/jax
    variant: returns a 1D int32 array of indices where condition is nonzero.
    """
    idx = mx.arange(condition.shape[0], dtype=mx.int32)
    # Use cumsum to compact
    mask = condition.astype(mx.int32)
    prefix = mx.cumsum(mask) - mask  # exclusive prefix sum
    n = mx.sum(mask)
    mx.eval(n)
    n_int = int(n.item())
    if n_int == 0:
        return mx.array([], dtype=mx.int32)
    # Scatter valid indices to compacted positions
    out = mx.zeros(n_int, dtype=mx.int32)
    safe_prefix = mx.where(condition, prefix, mx.zeros_like(prefix))
    out = out.at[safe_prefix].add(mx.where(condition, idx - out[safe_prefix], mx.zeros_like(idx)))
    return out


def searchsorted(sorted_arr: mx.array, values: mx.array, side: str = "left") -> mx.array:
    """Binary search — finds insertion indices in a sorted array.

    Pure MLX implementation since mx.searchsorted doesn't exist.
    Returns int32 indices such that sorted_arr[i-1] < value <= sorted_arr[i] (side='left')
    or sorted_arr[i-1] <= value < sorted_arr[i] (side='right').
    """
    import numpy as np
    # Materialize to numpy for the search, then return as mx array
    mx.eval(sorted_arr, values)
    s_np = np.array(sorted_arr)
    v_np = np.array(values)
    result = np.searchsorted(s_np, v_np, side=side).astype(np.int32)
    return mx.array(result)


def div_ceil(a, b):
    """Integer ceiling division."""
    return (a + b - 1) // b


def cumsum_starting_with_zero(x):
    """Cumulative sum with a leading zero."""
    return mx.concatenate([mx.zeros(1, dtype=x.dtype), mx.cumsum(x)])


def offset_sum(num):
    """Returns (offsets, total) from counts."""
    cs = mx.cumsum(num, axis=0)
    return cs - num, cs[-1]


def masked_prefix_sum(mask):
    """Prefix sum over a boolean mask, returning (offsets, total)."""
    mask_int = mask.astype(mx.int32)
    off, n = offset_sum(mask_int)
    off_masked = mx.where(mask, off, len(mask))
    return off_masked, n


def inverse_of_splits(ispl, size):
    """Given splits [0, 4, 7], returns segment indices [0,0,0,0,1,1,1] for size=7."""
    mask = mx.zeros(size, dtype=mx.int32)
    mask = mask.at[ispl].add(mx.ones_like(ispl))
    return mx.cumsum(mask) - 1


def inverse_indices(iargsort):
    """Given argsort indices, return the inverse permutation."""
    iunsort = mx.zeros_like(iargsort)
    iunsort = iunsort.at[iargsort].add(
        mx.arange(len(iargsort), dtype=iargsort.dtype)
    )
    return iunsort


def masked_inverse(indices, mask):
    """Given sort indices and a mask, return the inverse permutation."""
    inverse = mx.full(indices.shape, len(indices), dtype=indices.dtype)
    imask = mx.where(mask, indices, len(indices))
    arange = mx.arange(len(indices), dtype=indices.dtype)
    inverse = inverse.at[imask].add(arange - inverse[imask])  # effectively set
    # Simpler: just do scatter
    inverse = mx.full(indices.shape, len(indices), dtype=indices.dtype)
    for i in range(0, len(indices), 65536):  # batch to avoid huge intermediates
        end = min(i + 65536, len(indices))
        batch_mask = mask[i:end]
        batch_idx = indices[i:end]
        batch_vals = mx.arange(i, end, dtype=indices.dtype)
        valid_idx = mx.where(batch_mask, batch_idx, 0)
        inverse = mx.where(
            batch_mask.reshape(-1, 1) & (mx.arange(len(indices)).reshape(1, -1) == valid_idx.reshape(-1, 1)),
            batch_vals.reshape(-1, 1),
            inverse.reshape(1, -1)
        ).max(axis=0) if end - i < 1000 else inverse  # fallback for large
    # Actually, let's use the simple scatter approach
    inverse = mx.full(indices.shape, len(indices), dtype=indices.dtype)
    valid_idx = mx.where(mask, indices, len(indices))
    vals = mx.arange(len(indices), dtype=indices.dtype)
    inverse = inverse.at[valid_idx].add(vals - inverse[valid_idx])
    return inverse


def masked_scatter(mask, arr, indices, values):
    """Scatter values into arr at indices where mask is True."""
    safe_indices = mx.where(mask, indices, len(arr) - 1)
    # Build output: start with arr, then overwrite at valid locations
    result = arr * 1  # copy
    result = result.at[safe_indices].add(
        mx.where(mask, values - result[safe_indices], mx.zeros_like(values))
    )
    return result


def set_range(arr, values, start, end):
    """Set arr[start:end] = values[:end-start]."""
    idx = mx.arange(len(values)) + start
    valid = idx < end
    safe_idx = mx.where(valid, idx, 0)
    # Use where-based update
    arange_full = mx.arange(len(arr))
    in_range = (arange_full >= start) & (arange_full < end)
    in_range = in_range.reshape((-1,) + (1,) * (values.ndim - 1))
    return mx.where(in_range, values[arange_full - start], arr)


def bucket_prefix_sum(key, count=None, num=None):
    """Prefix sum per key: counts how many items with same key appeared before."""
    if num is not None:
        key = mx.where(mx.arange(len(key)) < num, key, 2**30)

    isort = mx.argsort(key)
    key_sort = key[isort]

    if count is None:
        csum_sort = mx.arange(len(key), dtype=key.dtype)
    else:
        count_sort = count[isort]
        csum_sort = mx.cumsum(count_sort) - count_sort

    # searchsorted to find first occurrence
    ifirst = searchsorted(key_sort, key_sort, side="left")
    cdiff = csum_sort - csum_sort[ifirst]

    if num is not None:
        isort = mx.where(mx.arange(len(isort)) < num, isort, len(isort) - 1)

    invsort = mx.zeros(len(isort), dtype=mx.int32)
    invsort = invsort.at[isort].add(mx.arange(len(isort), dtype=mx.int32))

    return cdiff[invsort]


def masked_to_dense(arr, mask, fill_value=0):
    """Compact masked elements to the front of the array."""
    mask_int = mask.astype(mx.int32)
    pref = mx.cumsum(mask_int) - mask_int  # exclusive prefix sum
    num = mx.sum(mask_int)
    pref = mx.where(mask, pref, len(mask))

    if isinstance(arr, mx.array):
        out = mx.full(arr.shape, fill_value, dtype=arr.dtype)
        safe_pref = mx.where(mask, pref, 0)
        # Scatter valid elements to compacted positions
        out = out.at[safe_pref].add(
            mx.where(mask.reshape(-1, *([1] * (arr.ndim - 1))),
                     arr - out[safe_pref], mx.zeros_like(arr))
        )
        return out, num
    else:
        # Handle tuple/list of arrays
        results = []
        for a in arr:
            r, _ = masked_to_dense(a, mask, fill_value)
            results.append(r)
        return tuple(results), num
