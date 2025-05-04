from abc import ABC, abstractmethod
from collections import OrderedDict
import re
from typing import Any, Dict, List
from visionflow.core.entity.reflection.meta import FieldMeta
from visionflow.core.inference.ocr.base import OcrResult
from visionflow.core.pipeline.base import Exchange


class FieldProcessorBase(ABC):
    @abstractmethod
    def process(self, field: FieldMeta, exchange: Exchange) -> Any:
        pass


class OcrRegexProcessor(FieldProcessorBase):
    def _group_by_line(self, ocr_results: List[OcrResult]) -> Dict[int, List[str]]:
        lines: Dict[str, List[str]] = OrderedDict()
        line_count = 0
        for r in sorted(ocr_results, key=lambda r: (r.page, r.paragraph, r.line, r.word)):
            lineno = r.line
            if lineno < line_count:
                line_count += 1
                lineno = line_count
            lines.setdefault(lineno, []).append(r.text)
        return lines
    
    def _results_to_lines(self, ocr_results: List[OcrResult]) -> List[str]:
        lines = self._group_by_line(ocr_results)
        return [" ".join(w for w in l) for l in lines.values()]

    def process(self, meta: FieldMeta, exchange: Exchange) -> Any:
        lines = self._results_to_lines(exchange.ocr_results)
        match_gen = None
        if meta.config.line_match:
            match_gen = (m.groupdict() for text in lines for m in re.finditer(meta.config.pattern, text))
        else:
            match_gen = (m.groupdict() for m in re.finditer(meta.config.pattern, "\n".join(lines)))
        
        if meta.config.stop_on_first_match:
            gd = next(match_gen, None)
            return gd if gd is None else meta.converter([gd])
        
        return [gd for gd in match_gen]

class EntityRefProcessor(FieldProcessorBase):
    def process(self, field: FieldMeta, exchange: Exchange) -> Any:
        return None

class ClassificationLabelProcessor(FieldProcessorBase):
    def process(self, meta: FieldMeta, exchange: Exchange) -> Any:
        labels = [c.label for c in exchange.classifications]
        if meta.config.allowed_labels:
            labels = [l for l in labels if l in meta.config.allowed_labels]
        return meta.converter(labels) if labels else None
