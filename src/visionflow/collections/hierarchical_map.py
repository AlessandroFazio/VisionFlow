from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar

from visionflow.core.collections.types import NodeBase, SupportsHash
from visionflow.core.collections.utils import GraphUtils

K = TypeVar("K", bound=SupportsHash)
V = TypeVar("V")


@dataclass
class Node(NodeBase[V]):
    parents: List["Node[V]"] = field(default_factory=list)


class HierarchicalMap(Generic[K, V]):
    def __init__(
        self,
        hierarchy: V,
        key_fn: Callable[[V], K],
        children_fn: Callable[[V], Iterable[V]],
        value_factory: Optional[Callable[[V], V]] = None
    ):
        self.key_fn = key_fn
        self.node_factory = value_factory or (lambda x: x)
        self.children_fn = children_fn
        self._key_map: Dict[K, Node[V]] = {}
        self._root = self._build_tree(hierarchy)

    def _build_tree(self, hierarchy: V) -> Node[V]:
        
        def build(h: V) -> Node[V]:
            key = self.key_fn(h)
            node = self._key_map.get(key)
            if node:
                return node 
            
            value = self.node_factory(h)
            node = Node(value)
            self._key_map[key] = node

            for child_h in self.children_fn(h):
                child_node = self._build_tree(child_h)
                child_node.parents.append(node)
                node.add_child(child_node)

            return node

        root = build(hierarchy)
        if GraphUtils.detect_cycles(self._root, self.key_fn):
            raise ValueError("")
        
        return root

    def root(self) -> Optional[V]:
        return self._root.value if not self.empty() else None
    
    def children_of(self, key: K) -> List[V]:
        node = self._key_map.get(key)
        return [child.value for child in node.children] if node else []
    
    def parents_of(self, key: K) -> List[V]:
        node = self._key_map.get(key)
        return [parent.value for parent in node.parents] if node else []

    def get(self, key: K) -> Optional[V]:
        node = self._key_map.get(key)
        return node.value if node else None

    def all(self) -> List[V]:
        return list((n.value for n in self._key_map.values()))

    def empty(self) -> bool:
        return self._root is None
    
    def __iter__(self) -> Iterator[Tuple[K, V]]:
        if not self.empty():
            yield from GraphUtils.iter_levels(self._root)

    def __contains__(self, key: K) -> bool:
        return key in self._key_map