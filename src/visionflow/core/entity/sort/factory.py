from typing import List
from visionflow.core.entity.sort.base import EntitySortStrategy
from visionflow.core.entity.sort.config import SortConfig, SortKey, SortType
from visionflow.core.entity.sort.strategies import ClockwiseSortStrategy, LexSortStrategy


class EntitySortFactory:
    @staticmethod
    def create(sort_type: SortType, sort_keys: List[SortKey]) -> EntitySortStrategy:
        if sort_type == SortType.CLOCKWISE:
            return ClockwiseSortStrategy(pivot_keys=sort_keys)
        elif sort_type == SortType.LEX:
            return LexSortStrategy(keys=sort_keys)
        else:
            raise ValueError("")