from typing import Iterable, Optional
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.parent_selector.base import ParentSelectorBase


class NearestParentSelectorStrategy(ParentSelectorBase):
    def select(self, e: EntityBase, candidates: Iterable[EntityBase]) -> Optional[EntityBase]:
        e_xy = e.coordinates()
        if not e_xy:
            return None
        ex, ey = e_xy
        best_c, best_d = None, float("inf")
        for p in candidates:
            c_xy = p.coordinates()
            if not c_xy:
                continue
            cx, cy = c_xy
            d = (ex - cx) ** 2 + (ey - cy) ** 2
            if d < best_d:
                best_c, best_d = p, d
        return best_c
    

class ExactParentSelectorStrategy(ParentSelectorBase):
    def select(self, e: EntityBase, candidates: Iterable[EntityBase]) -> Optional[EntityBase]:
        e_xy = e.coordinates()
        if not e_xy:
            return None
        for c in candidates:
            c_xy = c.coordinates()
            if not c_xy:
                continue
            if e_xy == c_xy:
                return c
        return None