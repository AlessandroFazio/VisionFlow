from typing import Set, Type

from visionflow.core.entity.base import EntityBase


class EntityUtils:

    @staticmethod
    def detect_cycle(entity_cls: Type[EntityBase]) -> bool:
        def _detect_cycle(entity_cls: Type[EntityBase], stack: Set[str]) -> bool:
                name = entity_cls.__name__
                if name in stack:
                    return True
                stack.add(name)
                meta = entity_cls.__vf_meta__
                for child_cls in meta.iter_children():
                    child_name = child_cls.__name__
                    if _detect_cycle(child_name, stack):
                        return True
                stack.remove(name)
                return False
        
        return _detect_cycle(entity_cls, set())