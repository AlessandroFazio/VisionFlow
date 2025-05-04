from dataclasses import dataclass, field
from typing import Generic, List, Optional, Protocol, TypeVar, runtime_checkable

T = TypeVar('T')


@runtime_checkable
class SupportsHash(Protocol):
    def __hash__(self) -> int: ...


@dataclass
class NodeBase(Generic[T]):
    value: T
    children: List["NodeBase[T]"] = field(default_factory=list)

    def add_child(self, child: "NodeBase[T]") -> None:
        self.children.append(child)