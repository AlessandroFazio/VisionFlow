from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.sort.config import SortKey

KeyFunction = Callable[[Tuple[float, float], Tuple[float, float]], Tuple[float, ...]]


class KeyFunctions:
    _functions: Dict[str, KeyFunction] = {
        SortKey.X_MIN: lambda _,p: p[0],
        SortKey.X_MAX: lambda _,p: -p[0],
        SortKey.Y_MIN: lambda _,p: p[1],
        SortKey.Y_MAX: lambda _,p:-p[1],
        SortKey.X_AVG: lambda c,p:abs(p[0] - c[0]),
        SortKey.Y_AVG: lambda c,p:abs(p[1] - c[1])
    }
    
    @classmethod
    def function(cls, keys: List[SortKey]) -> KeyFunction:
        functions = (cls._functions.get(k) for k in keys)
        return lambda c,p: tuple(f(c,p) for f in functions)


class EntitySortStrategy(ABC):
    @abstractmethod
    def argsort(self, entities: List[EntityBase]) -> List[int]:
        pass