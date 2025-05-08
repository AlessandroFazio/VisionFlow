from typing import Union

from visionflow.core.inference.base import ModelProvider
from visionflow.core.inference.classification.base import ClassificationModelBase
from visionflow.core.inference.config import InferenceType, RoboflowModelConfig, TesseractModelConfig
from visionflow.core.inference.detection.base import DetectionModelBase
from visionflow.core.inference.factory import ModelFactory
from visionflow.core.inference.ocr.base import OcrModelBase


class ModelBuilder:
    def __init__(self) -> None:
        self._inference_type = None
        self._provider = None
        self._config = None

    def inference(self, inference_type: Union[str, InferenceType]) -> "ModelBuilder":
        self._inference_type = (
            InferenceType.from_value(inference_type) if isinstance(inference_type, str)
            else inference_type
        )
        return self
    
    def provider(self, provider: ModelProvider) -> "ModelBuilder":
        self._provider = (
            ModelProvider.from_value(provider) if isinstance(provider, str)
            else provider
        )
        return self
    
    def config(self, config: Union[RoboflowModelConfig | TesseractModelConfig]) -> "ModelBuilder":
        self._config = config
        return self
        
    def build(self) -> Union[DetectionModelBase | OcrModelBase | ClassificationModelBase]:
        required = ["_provider", "_inference", "_config"]
        for attr in required:
            if not getattr(self, attr, None):
                raise ValueError("")
        return ModelFactory.create(**self._config)