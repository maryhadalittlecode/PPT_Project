from collections import deque
import numpy as np


def hysteresis(strong_map, weak_map, connectivity=8):
    """
    Sequential BFS hysteresis.

    Start from strong-edge pixels and grow into neighboring weak-edge pixels.
    Any weak pixel connected to a strong pixel becomes a final edge.
    """
    h, w = strong_map.shape
    result = strong_map.copy()
    visited = strong_map.copy()
    q = deque(zip(*np.nonzero(strong_map)))

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

    while q:
        r, c = q.popleft()
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if weak_map[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    result[nr, nc] = True
                    q.append((nr, nc))

    return result
