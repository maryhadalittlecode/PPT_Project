import numpy as np
import time
from numba import njit

@njit(cache=True)
def _hysteresis_frontier_kernel(strong_map, weak_map, connectivity):
    h, w = strong_map.shape
    result = strong_map.copy()

    # Current frontier: pixels we just activated and need to check neighbors for
    # We use a boolean mask for the 'next' frontier to ensure uniqueness
    current_frontier = np.argwhere(strong_map)

    if connectivity == 4:
        neighbors = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
    else:
        neighbors = np.array([
            [-1, 0], [1, 0], [0, -1], [0, 1],
            [-1, -1], [-1, 1], [1, -1], [1, 1]
        ], dtype=np.int32)

    while len(current_frontier) > 0:
        # A boolean mask for the next level to prevent redundant neighbor checks
        next_mask = np.zeros((h, w), dtype=np.bool_)
        found_any = False

        for i in range(len(current_frontier)):
            r = current_frontier[i, 0]
            c = current_frontier[i, 1]

            for j in range(len(neighbors)):
                nr, nc = r + neighbors[j, 0], c + neighbors[j, 1]

                if 0 <= nr < h and 0 <= nc < w:
                    # If it's a weak edge and not yet part of our result
                    if weak_map[nr, nc] and not result[nr, nc]:
                        result[nr, nc] = True
                        next_mask[nr, nc] = True
                        found_any = True

        if not found_any:
            break

        current_frontier = np.argwhere(next_mask)

    return result

def hysteresis(strong_map, weak_map, connectivity=8):
    """
    Frontier-based BFS Hysteresis (Level-Synchronous).
    Optimized with Numba.
    """
    # Ensure contiguous arrays for Numba speed
    strong_map = np.ascontiguousarray(strong_map.astype(np.bool_))
    weak_map = np.ascontiguousarray(weak_map.astype(np.bool_))

    start = time.perf_counter()
    result = _hysteresis_frontier_kernel(strong_map, weak_map, connectivity)
    elapsed = time.perf_counter() - start

    return result, elapsed