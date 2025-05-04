from collections import deque
from typing import Callable, Iterator, Set

from visionflow.core.collections.types import T, NodeBase, SupportsHash


class GraphUtils:
    @staticmethod
    def detect_cycles(start: NodeBase[T], key_fn: Callable[[T], SupportsHash]) -> bool:
        def _detect_cycle(node: NodeBase[T], stack: Set[str]) -> bool:
            key = key_fn(node.value)
            if key in stack:
                return True
            
            stack.add(key)
            for child in node.children:
                if _detect_cycle(child, stack):
                    return True
            
            stack.remove(key)
            return False
        
        return _detect_cycle(start, set())
    
    @staticmethod
    def iter_levels(start: NodeBase[T]) -> Iterator[T]:
        queue = deque([[start]])
        while queue:
            current_lvl = queue.popleft()
            yield from current_lvl
            next_lvl = [child for node in current_lvl for child in node]
            if next_lvl:
                queue.appendleft(next_lvl)
