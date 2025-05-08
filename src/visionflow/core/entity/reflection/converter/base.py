from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar, Union

from visionflow.core.entity.base import EntityBase
from visionflow.core.regex_matcher.base import RegexMatchResult

I = TypeVar('I')
O = TypeVar('O')
EntityOrEntityList = Union[EntityBase | List[EntityBase]]


class TypeConverterBase(ABC, Generic[I]):
    @abstractmethod
    def convert(self, input: I) -> O:
        pass
    
    def __call__(self, input: I) -> O:
        return self.convert(input)


class RegexMatcherConverterBase(TypeConverterBase[List[RegexMatchResult]], Generic[O]):
    pass


class ClassificationLabelConverterBase(TypeConverterBase[List[str]], Generic[O]):
    pass


class EntityRefConverterBase(TypeConverterBase[EntityOrEntityList], Generic[O]):
    pass


ConverterBases = Union[RegexMatcherConverterBase[O], EntityRefConverterBase[O], ClassificationLabelConverterBase[O]]