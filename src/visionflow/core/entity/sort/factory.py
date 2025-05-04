from typing import Any, Dict
from visionflow.core.entity.sort.base import EntitySortStrategy
from visionflow.core.entity.sort.config import SortConfig, SortType
from visionflow.core.entity.sort.strategies import ClockwiseSortStrategy, LexSortStrategy


class EntitySortFactory:
    @staticmethod
    def create(config: SortConfig) -> EntitySortStrategy:
        if config.type == SortType.CLOCKWISE:
            return ClockwiseSortStrategy(pivot_keys=config.keys)
        elif config.type == SortType.LEX:
            return LexSortStrategy(keys=config.keys)
        else:
            raise ValueError("")