from typing import Any
from visionflow.core.entity.reflection.descriptor.base import FieldProcessorBase
from visionflow.core.entity.reflection.meta.fields import FieldMeta
from visionflow.core.pipeline.base import Exchange


class OcrRegexProcessor(FieldProcessorBase):
    def process(self, meta: FieldMeta, exchange: Exchange) -> Any:
        matches = exchange.ocr_regex_matches
        selector = meta.config.rule_selector
        match_key = meta.config.match_key
        if selector:
            matches = [m for m in matches if m.rule_id in selector]
        if match_key:
            matches = [{match_key: m[match_key]} for m in matches if match_key in m]
        if not len(matches):
            return None
        return meta.converter(matches)


class EntityRefProcessor(FieldProcessorBase):
    def process(self, field: FieldMeta, exchange: Exchange) -> Any:
        return None


class ClassificationLabelProcessor(FieldProcessorBase):
    def process(self, meta: FieldMeta, exchange: Exchange) -> Any:
        labels = [c.label for c in exchange.classifications]
        if meta.config.allowed_labels:
            labels = [l for l in labels if l in meta.config.allowed_labels]
        return meta.converter(labels) if labels else None
