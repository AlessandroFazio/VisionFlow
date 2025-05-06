from typing import Iterator, Type

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.utils import Entity


class EntityHierarchyIterator:
    def __init__(self, root: Type[EntityBase]):
        self._root = root

    def __iter__(self) -> Iterator[Type[EntityBase]]:
        seen = set()

        def dfs(cls: Type[EntityBase]) -> Iterator[Type[EntityBase]]:
            name = Entity.name(cls)
            if name in seen:
                return
            seen.add(name)
            yield cls
            for _, child in Entity.iter_children(cls):
                child_cls = child.type_info.base_type
                yield from dfs(child_cls)

        return dfs(self._root)