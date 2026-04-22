import os
import numpy as np
import time

try:
    from numba import njit, set_num_threads
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False

@njit(cache=True)
def _find(parent, i):
    # Path halving: a faster version of path compression for Numba
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

@njit(cache=True)
def _union(parent, rank, has_strong, i, j):
    root_i = _find(parent, i)
    root_j = _find(parent, j)
    if root_i != root_j:
        # Union by rank to keep trees flat
        if rank[root_i] < rank[root_j]:
            parent[root_i] = root_j
            if has_strong[root_i]:
                has_strong[root_j] = True
        else:
            parent[root_j] = root_i
            if has_strong[root_j]:
                has_strong[root_i] = True
            if rank[root_i] == rank[root_j]:
                rank[root_i] += 1

@njit(cache=True)
def _hysteresis_uf_kernel(strong_u8, weak_u8, connectivity):
    h, w = strong_u8.shape
    num_pixels = h * w

    parent = np.arange(num_pixels, dtype=np.int32)
    rank = np.zeros(num_pixels, dtype=np.uint16)

    # Flattening for faster access
    strong_flat = strong_u8.ravel()
    weak_flat = weak_u8.ravel()
    has_strong = (strong_flat > 0)
    is_edge = (strong_flat > 0) | (weak_flat > 0)

    # Neighborhood offsets
    offsets = [(0, 1), (1, 0)]
    if connectivity == 8:
        offsets.extend([(1, 1), (1, -1)])

    # Only process pixels that are part of an edge
    for r in range(h):
        for c in range(w):
            idx = r * w + c
            if is_edge[idx]:
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        n_idx = nr * w + nc
                        if is_edge[n_idx]:
                            _union(parent, rank, has_strong, idx, n_idx)

    # Final pass: check roots
    result = np.zeros(num_pixels, dtype=np.bool_)
    for i in range(num_pixels):
        if is_edge[i]:
            if has_strong[_find(parent, i)]:
                result[i] = True

    return result.reshape((h, w))

def hysteresis(strong_map, weak_map, connectivity=8):
    """
    Optimized Union-Find Hysteresis
    """
    strong_u8 = np.ascontiguousarray((np.asarray(strong_map) != 0).astype(np.uint8))
    weak_u8 = np.ascontiguousarray((np.asarray(weak_map) != 0).astype(np.uint8))

    start = time.perf_counter()
    result = _hysteresis_uf_kernel(strong_u8, weak_u8, connectivity)
    elapsed = time.perf_counter() - start

    return result, elapsed