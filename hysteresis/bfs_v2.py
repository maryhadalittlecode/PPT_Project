import os
import numpy as np
import time

try:
    from numba import njit, prange, set_num_threads
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _expand_frontier_numba(frontier, connectivity, expanded):
        """
        Expand the current frontier by one BFS level.

        expanded[r, c] = 1 iff pixel (r, c) is adjacent to the current frontier.
        """
        h, w = frontier.shape

        for r in prange(h):
            for c in range(w):
                v = 0

                # 4-neighborhood
                if r > 0 and frontier[r - 1, c] != 0:
                    v = 1
                elif r + 1 < h and frontier[r + 1, c] != 0:
                    v = 1
                elif c > 0 and frontier[r, c - 1] != 0:
                    v = 1
                elif c + 1 < w and frontier[r, c + 1] != 0:
                    v = 1
                elif connectivity == 8:
                    # diagonal neighbors
                    if r > 0 and c > 0 and frontier[r - 1, c - 1] != 0:
                        v = 1
                    elif r > 0 and c + 1 < w and frontier[r - 1, c + 1] != 0:
                        v = 1
                    elif r + 1 < h and c > 0 and frontier[r + 1, c - 1] != 0:
                        v = 1
                    elif r + 1 < h and c + 1 < w and frontier[r + 1, c + 1] != 0:
                        v = 1

                expanded[r, c] = v


    @njit(parallel=True, cache=True)
    def _activate_frontier_numba(expanded, weak_remaining, result, frontier_out):
        """
        Keep only newly reached weak pixels.

        Returns
        -------
        activated : int
            Number of new pixels activated in this round.
        """
        h, w = expanded.shape
        row_counts = np.zeros(h, dtype=np.int32)

        for r in prange(h):
            local_count = 0
            for c in range(w):
                is_new = 1 if (expanded[r, c] != 0 and weak_remaining[r, c] != 0) else 0
                frontier_out[r, c] = is_new

                if is_new:
                    weak_remaining[r, c] = 0
                    result[r, c] = 1
                    local_count += 1

            row_counts[r] = local_count

        activated = 0
        for r in range(h):
            activated += row_counts[r]

        return activated


    @njit(cache=True)
    def _hysteresis_frontier_numba(strong, weak, connectivity):
        """
        Frontier-based BFS hysteresis using level-synchronous expansion.
        """
        h, w = strong.shape

        frontier_in = strong.copy()
        frontier_out = np.zeros((h, w), dtype=np.uint8)

        result = strong.copy()
        weak_remaining = weak.copy()

        # remove strong pixels from weak candidates
        for r in range(h):
            for c in range(w):
                if strong[r, c] != 0:
                    weak_remaining[r, c] = 0

        expanded = np.zeros((h, w), dtype=np.uint8)

        while True:
            # Step 1: expand current frontier
            _expand_frontier_numba(frontier_in, connectivity, expanded)

            # Step 2: activate only newly reached weak pixels
            activated = _activate_frontier_numba(
                expanded, weak_remaining, result, frontier_out
            )

            # Step 3: stop if no new pixels are found
            if activated == 0:
                break

            # Step 4: swap frontiers
            frontier_in, frontier_out = frontier_out, frontier_in

        return result


def _expand_frontier_numpy(frontier, connectivity, expanded):
    """
    NumPy fallback: expand the current frontier by one BFS level.
    """
    expanded.fill(False)

    # 4-neighborhood
    expanded[1:, :] |= frontier[:-1, :]
    expanded[:-1, :] |= frontier[1:, :]
    expanded[:, 1:] |= frontier[:, :-1]
    expanded[:, :-1] |= frontier[:, 1:]

    if connectivity == 8:
        expanded[1:, 1:] |= frontier[:-1, :-1]
        expanded[1:, :-1] |= frontier[:-1, 1:]
        expanded[:-1, 1:] |= frontier[1:, :-1]
        expanded[:-1, :-1] |= frontier[1:, 1:]

    return expanded


def _hysteresis_frontier_numpy(strong, weak, connectivity):
    """
    NumPy fallback frontier BFS hysteresis.
    """
    frontier_in = strong.copy()
    result = strong.copy()
    weak_remaining = weak & (~strong)
    expanded = np.zeros_like(frontier_in, dtype=bool)

    while frontier_in.any():
        _expand_frontier_numpy(frontier_in, connectivity, expanded)
        frontier_out = expanded & weak_remaining

        if not frontier_out.any():
            break

        result |= frontier_out
        weak_remaining &= ~frontier_out
        frontier_in = frontier_out

    return result


def hysteresis(strong_map, weak_map, connectivity=8):
    """
    Frontier-based Canny hysteresis thresholding.

    Parameters
    ----------
    strong_map : array-like
        Nonzero means strong edge.
    weak_map : array-like
        Nonzero means weak edge.
    connectivity : int
        4 or 8.

    Returns
    -------
    edges : np.ndarray, bool
        Final hysteresis output.
    elapsed : float
        Runtime in seconds.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    strong = np.ascontiguousarray((np.asarray(strong_map) != 0).astype(np.uint8))
    weak = np.ascontiguousarray((np.asarray(weak_map) != 0).astype(np.uint8))

    if strong.shape != weak.shape:
        raise ValueError("strong_map and weak_map must have the same shape")

    start = time.perf_counter()

    if HAS_NUMBA:
        threads = os.getenv("HYST_NUM_THREADS")
        if threads:
            set_num_threads(max(1, int(threads)))
        edges = _hysteresis_frontier_numba(strong, weak, connectivity).astype(bool)
    else:
        edges = _hysteresis_frontier_numpy(strong.astype(bool), weak.astype(bool), connectivity)

    elapsed = time.perf_counter() - start
    return edges, elapsed


if __name__ == "__main__":
    strong = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)

    weak = np.array([
        [1, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ], dtype=np.uint8)

    # Warmup to exclude first-call JIT compile overhead
    if HAS_NUMBA:
        _ = hysteresis(strong, weak, connectivity=8)

    out4, t4 = hysteresis(strong, weak, connectivity=4)
    out8, t8 = hysteresis(strong, weak, connectivity=8)

    print("Strong:\n", strong)
    print("Weak:\n", weak)
    print("Result (4-connectivity):\n", out4.astype(np.uint8))
    print(f"Elapsed (4-connectivity): {t4:.6f} s")
    print("Result (8-connectivity):\n", out8.astype(np.uint8))
    print(f"Elapsed (8-connectivity): {t8:.6f} s")
