from typing import Type, TypeVar, Union


T = TypeVar('T')
U = TypeVar('U')

class ClassFactory:
    @staticmethod
    def ensure_base(cls: Type[T], base: Type[U]) -> Type[Union[U, T]]:
        new_cls = type(cls.__name__, (cls, base), dict(cls.__dict__))
        new_cls.__module__ = cls.__module__
        new_cls.__qualname__ = cls.__qualname__
        return new_cls