import os
import numpy as np
import time

try:
    from numba import njit, set_num_threads
    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False


# =========================
# Numba-accelerated kernels
# =========================

if HAS_NUMBA:
    @njit(cache=True)
    def _seed_queue_numba(strong, weak_remaining, result, queue_r, queue_c):
        """
        Initialize the BFS queue with all strong pixels.
        Also clears any overlap from weak_remaining and writes strong pixels into result.

        Returns:
            tail: number of queued items
        """
        h, w = strong.shape
        tail = 0
        for r in range(h):
            for c in range(w):
                if strong[r, c] != 0:
                    result[r, c] = 1
                    weak_remaining[r, c] = 0
                    queue_r[tail] = r
                    queue_c[tail] = c
                    tail += 1
        return tail


    @njit(cache=True)
    def _hysteresis_bfs_numba(strong, weak, connectivity):
        """
        Work-efficient hysteresis using a queue-based BFS.

        This avoids rescanning the full image each propagation round.
        Every strong pixel is used as a source. Weak pixels are promoted exactly once
        if they are reachable from any strong pixel under the chosen connectivity.
        """
        h, w = strong.shape

        result = np.zeros((h, w), dtype=np.uint8)
        weak_remaining = weak.copy()

        # Worst case: every pixel gets queued once.
        queue_r = np.empty(h * w, dtype=np.int32)
        queue_c = np.empty(h * w, dtype=np.int32)

        head = 0
        tail = _seed_queue_numba(strong, weak_remaining, result, queue_r, queue_c)

        while head < tail:
            r = queue_r[head]
            c = queue_c[head]
            head += 1

            # 4-neighborhood
            nr = r - 1
            nc = c
            if nr >= 0 and weak_remaining[nr, nc] != 0:
                weak_remaining[nr, nc] = 0
                result[nr, nc] = 1
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

            nr = r + 1
            nc = c
            if nr < h and weak_remaining[nr, nc] != 0:
                weak_remaining[nr, nc] = 0
                result[nr, nc] = 1
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

            nr = r
            nc = c - 1
            if nc >= 0 and weak_remaining[nr, nc] != 0:
                weak_remaining[nr, nc] = 0
                result[nr, nc] = 1
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

            nr = r
            nc = c + 1
            if nc < w and weak_remaining[nr, nc] != 0:
                weak_remaining[nr, nc] = 0
                result[nr, nc] = 1
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1

            # 8-neighborhood diagonals
            if connectivity == 8:
                nr = r - 1
                nc = c - 1
                if nr >= 0 and nc >= 0 and weak_remaining[nr, nc] != 0:
                    weak_remaining[nr, nc] = 0
                    result[nr, nc] = 1
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

                nr = r - 1
                nc = c + 1
                if nr >= 0 and nc < w and weak_remaining[nr, nc] != 0:
                    weak_remaining[nr, nc] = 0
                    result[nr, nc] = 1
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

                nr = r + 1
                nc = c - 1
                if nr < h and nc >= 0 and weak_remaining[nr, nc] != 0:
                    weak_remaining[nr, nc] = 0
                    result[nr, nc] = 1
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

                nr = r + 1
                nc = c + 1
                if nr < h and nc < w and weak_remaining[nr, nc] != 0:
                    weak_remaining[nr, nc] = 0
                    result[nr, nc] = 1
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

        return result


# =========================
# NumPy fallback
# =========================

def _hysteresis_bfs_numpy(strong, weak, connectivity):
    """
    Pure NumPy/Python fallback.
    Slower than the numba path, but preserves the same semantics.
    """
    h, w = strong.shape
    result = np.zeros((h, w), dtype=np.uint8)
    weak_remaining = weak.copy()

    queue = []
    for r, c in zip(*np.nonzero(strong)):
        result[r, c] = 1
        weak_remaining[r, c] = 0
        queue.append((r, c))

    head = 0
    if connectivity == 8:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while head < len(queue):
        r, c = queue[head]
        head += 1

        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and weak_remaining[nr, nc] != 0:
                weak_remaining[nr, nc] = 0
                result[nr, nc] = 1
                queue.append((nr, nc))

    return result


# =========================
# Public API
# =========================

def hysteresis(strong_map, weak_map, connectivity=8):
    """
    Canny hysteresis thresholding.

    Args:
        strong_map: array-like, nonzero means strong edge.
        weak_map: array-like, nonzero means weak edge.
        connectivity: 4 or 8.

    Returns:
        bool array of final kept edges.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    strong_u8 = np.ascontiguousarray((np.asarray(strong_map) != 0).astype(np.uint8))
    weak_u8 = np.ascontiguousarray((np.asarray(weak_map) != 0).astype(np.uint8))

    if strong_u8.shape != weak_u8.shape:
        raise ValueError("strong_map and weak_map must have the same shape")

    assert HAS_NUMBA, "Numba is required for the frontier_parallel method. Please install numba or switch to a different method."
    threads = os.getenv("HYST_NUM_THREADS")
    if threads:
        set_num_threads(max(1, int(threads)))
    start = time.perf_counter()
    edges =  _hysteresis_bfs_numba(strong_u8, weak_u8, connectivity).astype(bool)
    elapsed = time.perf_counter() - start

    return edges, elapsed


# =========================
# Small self-test
# =========================

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

    out4 = hysteresis(strong, weak, connectivity=4)
    out8 = hysteresis(strong, weak, connectivity=8)

    print("Strong:\n", strong)
    print("Weak:\n", weak)
    print("Result (4-connectivity):\n", out4.astype(np.uint8))
    print("Result (8-connectivity):\n", out8.astype(np.uint8))



# Processed 11 image(s). Total hysteresis time: 0.220140 seconds