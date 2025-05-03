from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Type

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.utils import EntityUtils


@dataclass
class EntityRegistry:
    entities: Dict[str, Type[EntityBase]]
    parenthood: Dict[str, List[Type[EntityBase]]]

    @classmethod
    def from_root(cls, root: Type[EntityBase]) -> "EntityRegistry":
        entities = {}
        visited = set()
        parenthood = defaultdict(list)
        queue = deque([root])

        if EntityUtils.detect_cycle(root, set()):
            raise ValueError("Cycle in entity definitions")

        while queue:
            entity_cls = queue.popleft()
            name = entity_cls.__name__
            if name in visited:
                continue
            visited.add(name)
            entities[name] = entity_cls
            meta = entity_cls.__vf_meta__
            for child_cls in meta.iter_children():
                child_name = child_cls.__name__
                parenthood[child_name].append(entity_cls)
                queue.append(child_cls)

        return cls(entities) 