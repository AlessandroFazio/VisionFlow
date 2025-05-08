from typing import List, Optional, Type, Union

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.converter.base import ClassificationLabelConverterBase, EntityRefConverterBase, RegexMatcherConverterBase
from visionflow.core.entity.reflection.converter.defaults import DefaultClassificationLabelConverter, DefaultEntityRefConverter, DefaultRegexMatcherConverter
from visionflow.core.entity.reflection.descriptor.base import FieldBase
from visionflow.core.entity.sort.config import SortType


class ClassificationLabel(FieldBase):
    def __init__(
        self, 
        allowed_labels: List[str]=None, 
        converter: Optional[ClassificationLabelConverterBase]=None
    ) -> None:
        super().__init__()
        self.allowed_labels = allowed_labels
        self.converter = converter or DefaultClassificationLabelConverter()


class OcrRegex(FieldBase):
    def __init__(
        self,
        converter: Optional[RegexMatcherConverterBase]=None,
        rule_selector: Optional[List[str]]=None,
        match_key: Optional[str]=None, 
    ) -> None:
        super().__init__()
        self.converter = converter or DefaultRegexMatcherConverter(match_key)
        self.rule_selector = rule_selector
        self.match_key = match_key


class EntityRef(FieldBase):
    def __init__(
        self,
        entity_cls: Optional[Type[EntityBase]]=None,
        sort_type: Union[str | SortType | None]=None,
        sort_keys: Optional[List[str]]=None,
        converter: Optional[EntityRefConverterBase]=None
    ) -> None:
        super().__init__()
        self.entity_cls = entity_cls
        self.sort_type = sort_type
        self.sort_keys = sort_keys
        self.converter = converter or DefaultEntityRefConverter()
