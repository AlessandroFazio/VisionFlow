from enum import Enum
from typing import Any, TypeVar, Type

T = TypeVar("T", bound="ValueLookupMixin")


class EnumValueLookupMixin(Enum):
    @classmethod
    def from_value(cls: Type[T], value: Any) -> T:
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value!r} is not a valid value for {cls.__name__}")
