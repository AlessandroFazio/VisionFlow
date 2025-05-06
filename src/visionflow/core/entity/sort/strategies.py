import math
from typing import List, Tuple
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.sort.base import EntitySortStrategy, KeyFunctions


def _points_for_entities(entities: List[EntityBase]) -> List[Tuple[float, float]]:
    points = (e.coordinates(normalized=True) for e in entities)
    points = [p for p in points if p is not None]
    return points


class ClockwiseSortStrategy(EntitySortStrategy):
    def __init__(self, pivot_keys: List[str]):
        self.pivot_func = KeyFunctions.function(pivot_keys)

    def argsort(self, entities: List[EntityBase]) -> List[int]:
        points = _points_for_entities(entities)
        if not points or len(points) != len(entities):
            return []
        
        cx, cy = 0.5, 0.5
        pivot_func = lambda i: self.pivot_func((cx, cy), points[i])
        pivot = max(range(len(entities)), key=pivot_func)

        def angle(i):
            px, py = points[i]
            return (math.atan2(py - cy, px - cx) + 2 * math.pi) % (2 * math.pi)

        sorted_idx = sorted(range(len(points)), key=angle)
        pivot_pos = sorted_idx.index(pivot)
        return sorted_idx[pivot_pos:] + sorted_idx[:pivot_pos]


class LexSortStrategy(EntitySortStrategy):
    def __init__(self, keys: List[str]):
        self.key_func = KeyFunctions.function(keys)

    def argsort(self, entities: List[EntityBase]) -> List[int]:
        points = _points_for_entities(entities)
        if not points or len(points) != len(entities):
            return []

        cx = sum(x for x, _ in points) / len(points)
        cy = sum(y for _, y in points) / len(points)
        return sorted(
            range(len(points)),
            key=lambda i: self.key_func((cx,cy), points[i])
        )