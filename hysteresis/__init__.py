from .sequential_bfs import hysteresis as sequential_bfs
from .frontier_parallel import hysteresis as frontier_parallel
from .union_find import hysteresis as union_find

METHODS = {
    "sequential_bfs": sequential_bfs,
    "frontier_parallel": frontier_parallel,
    "union_find": union_find,
}
