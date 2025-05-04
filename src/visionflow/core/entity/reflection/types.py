from typing import Any, Callable, Dict, List, TypeVar

from visionflow.core.entity.base import EntityBase


TargetType = TypeVar('TargetType')
RegexGroupDictType = Dict[str, str | Any]
OcrRegexConverter = Callable[[List[RegexGroupDictType]], TargetType]
ClassificationLabelConverter = Callable[[List[str], TargetType]]
EntityRefConverter = Callable[[EntityBase], TargetType]

def default_single_label_converter(labels: List[str]) -> str:
    return labels[0]