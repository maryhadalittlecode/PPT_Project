import numpy as np
import time
from numba import njit


@njit(cache=True)
def _init_frontier(strong_map, frontier):
    """
    Fill frontier with coordinates of strong pixels.
    Returns frontier size.
    """
    h, w = strong_map.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if strong_map[r, c]:
                frontier[count, 0] = r
                frontier[count, 1] = c
                count += 1
    return count


@njit(cache=True)
def _hysteresis_frontier_kernel_4(strong_map, weak_map):
    h, w = strong_map.shape
    result = strong_map.copy()

    max_size = h * w

    # Preallocate coordinate-list frontiers
    frontier_cur = np.empty((max_size, 2), dtype=np.int32)
    frontier_next = np.empty((max_size, 2), dtype=np.int32)

    cur_size = _init_frontier(strong_map, frontier_cur)

    while cur_size > 0:
        next_size = 0
        next_mask = np.zeros((h, w), dtype=np.bool_)

        for i in range(cur_size):
            r = frontier_cur[i, 0]
            c = frontier_cur[i, 1]

            # up
            nr, nc = r - 1, c
            if nr >= 0:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            # down
            nr, nc = r + 1, c
            if nr < h:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            # left
            nr, nc = r, c - 1
            if nc >= 0:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            # right
            nr, nc = r, c + 1
            if nc < w:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

        frontier_cur, frontier_next = frontier_next, frontier_cur
        cur_size = next_size

    return result


@njit(cache=True)
def _hysteresis_frontier_kernel_8(strong_map, weak_map):
    h, w = strong_map.shape
    result = strong_map.copy()

    max_size = h * w

    # Preallocate coordinate-list frontiers
    frontier_cur = np.empty((max_size, 2), dtype=np.int32)
    frontier_next = np.empty((max_size, 2), dtype=np.int32)

    cur_size = _init_frontier(strong_map, frontier_cur)

    while cur_size > 0:
        next_size = 0
        next_mask = np.zeros((h, w), dtype=np.bool_)

        for i in range(cur_size):
            r = frontier_cur[i, 0]
            c = frontier_cur[i, 1]

            # 8 neighbors, manually unrolled

            nr, nc = r - 1, c
            if nr >= 0:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            nr, nc = r + 1, c
            if nr < h:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            nr, nc = r, c - 1
            if nc >= 0:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            nr, nc = r, c + 1
            if nc < w:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            nr, nc = r - 1, c - 1
            if nr >= 0 and nc >= 0:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            nr, nc = r - 1, c + 1
            if nr >= 0 and nc < w:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            nr, nc = r + 1, c - 1
            if nr < h and nc >= 0:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

            nr, nc = r + 1, c + 1
            if nr < h and nc < w:
                if weak_map[nr, nc] and (not result[nr, nc]) and (not next_mask[nr, nc]):
                    result[nr, nc] = True
                    next_mask[nr, nc] = True
                    frontier_next[next_size, 0] = nr
                    frontier_next[next_size, 1] = nc
                    next_size += 1

        frontier_cur, frontier_next = frontier_next, frontier_cur
        cur_size = next_size

    return result


def hysteresis(strong_map, weak_map, connectivity=8):
    strong_map = np.ascontiguousarray(strong_map.astype(np.bool_))
    weak_map = np.ascontiguousarray(weak_map.astype(np.bool_))

    start = time.perf_counter()

    if connectivity == 4:
        result = _hysteresis_frontier_kernel_4(strong_map, weak_map)
    elif connectivity == 8:
        result = _hysteresis_frontier_kernel_8(strong_map, weak_map)
    else:
        raise ValueError("connectivity must be 4 or 8")

    elapsed = time.perf_counter() - start
    return result, elapsed