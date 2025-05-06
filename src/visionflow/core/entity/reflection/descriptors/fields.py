from typing import List, Optional, Type, Union

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.descriptors.base import FieldBase
from visionflow.core.entity.reflection.types import ClassificationLabelConverter, EntityRefConverter, OcrRegexConverter, default_single_label_converter
from visionflow.core.entity.sort.config import SortType


class ClassificationLabel(FieldBase):
    def __init__(
        self, 
        allowed_labels: List[str]=None, 
        converter: Optional[ClassificationLabelConverter]=None
    ) -> None:
        super().__init__()
        self.allowed_labels = allowed_labels
        self.converter = converter or default_single_label_converter


class OcrRegex(FieldBase):
    def __init__(
        self,
        converter: OcrRegexConverter,
        rule_selector: Optional[List[str]]=None,
        match_key: Optional[str]=None, 
    ) -> None:
        super().__init__()
        self.converter = converter
        self.rule_selector = rule_selector
        self.match_key = match_key


class EntityRef(FieldBase):
    def __init__(
        self,
        entity_cls: Optional[Type[EntityBase]]=None,
        sort_type: Union[str | SortType | None]=None,
        sort_keys: Optional[List[str]]=None,
        converter: Optional[EntityRefConverter]=None
    ) -> None:
        super().__init__()
        self.entity_cls = entity_cls
        self.sort_type = sort_type
        self.sort_keys = sort_keys
        self.converter = converter or (lambda x: x)
