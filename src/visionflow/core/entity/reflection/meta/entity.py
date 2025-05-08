from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from visionflow.core.entity.parent_selector.base import ParentSelectorBase
from visionflow.core.entity.reflection.meta.fields import FieldMeta
from visionflow.core.pipeline.utils.providers import CoordinatesProviderBase


class EntityType(Enum):
    LOGICAL = "logical"
    PHYSICAL = "physical"


@dataclass
class EntitySource:
    type: EntityType
    provider: Optional[CoordinatesProviderBase] = None


@dataclass
class EntityMeta:
    source: EntitySource
    parent_selector: Optional[ParentSelectorBase] = None
    fields: Dict[str, FieldMeta] = field(default_factory=dict)