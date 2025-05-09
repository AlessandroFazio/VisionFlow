from typing import Any, Generic, TypeVar

T = TypeVar("T")


class ProxyMixin(Generic[T]):
    _wrapped: T

    def __init__(self, wrapped: T) -> None:
        self._wrapped = wrapped

    def __getattr__(self, name: str) -> Any:
        cls_attr = getattr(self.__class__, name, None)
        if callable(cls_attr):
            return cls_attr.__get__(self, self.__class__)
        return getattr(self._wrapped, name)

    def __dir__(self):
        return sorted(set(dir(type(self)) + dir(self._wrapped)))