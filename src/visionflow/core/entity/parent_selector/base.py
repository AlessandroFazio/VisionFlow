from abc import ABC, abstractmethod
from typing import Iterable, Optional

from visionflow.core.entity.base import EntityBase


class ParentSelectorBase(ABC):
    @abstractmethod
    def select(self, entity: EntityBase, candidates: Iterable[EntityBase]) -> Optional[EntityBase]:
        pass
