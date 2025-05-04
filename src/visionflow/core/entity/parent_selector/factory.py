from visionflow.core.entity.parent_selector.base import ParentSelectorBase
from visionflow.core.entity.parent_selector.strategies import NearestParentSelectorStrategy, ExactParentSelectorStrategy
from visionflow.core.entity.parent_selector.config import ParentSelectorType


class ParentSelectorFactory:
    @staticmethod
    def create(type: ParentSelectorType) -> ParentSelectorBase:
        if type == ParentSelectorType.NEAREST:
            return NearestParentSelectorStrategy()
        elif type == ParentSelectorType.EXACT:
            return ExactParentSelectorStrategy()
        raise ValueError("")