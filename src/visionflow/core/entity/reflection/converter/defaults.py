from typing import List
from visionflow.core.entity.reflection.converter.base import ClassificationLabelConverterBase, EntityOrEntityList, EntityRefConverterBase, RegexMatcherConverterBase
from visionflow.core.regex_matcher.base import RegexMatchResult


class DefaultRegexMatcherConverter(RegexMatcherConverterBase[str]):
    def __init__(self, match_key: str) -> None:
        self.match_key = match_key
        
    def convert(self, results: List[RegexMatchResult]) -> str:
        return results[0].matches.get(self.match_key)


class DefaultClassificationLabelConverter(ClassificationLabelConverterBase[str]):
    def convert(self, labels: List[str]) -> str:
        return labels[0]
    
    
class DefaultEntityRefConverter(EntityRefConverterBase[EntityOrEntityList]):
    def convert(self, entity: EntityOrEntityList) -> EntityOrEntityList:
        return entity
    

class EntityRefIsPresent(EntityRefConverterBase[bool]):
    def convert(self, entity: EntityOrEntityList) -> bool:
        return bool(entity)