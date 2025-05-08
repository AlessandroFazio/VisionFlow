from typing import Union
from visionflow.core.inference.classification.base import ClassificationModelBase
from visionflow.core.inference.classification.models import RoboflowClassificationModel
from visionflow.core.inference.config import ModelConfig, ModelProvider, InferenceType
from visionflow.core.inference.detection.base import DetectionModelBase
from visionflow.core.inference.detection.models import RoboflowDetectionModel
from visionflow.core.inference.ocr.base import OcrModelBase
from visionflow.core.inference.ocr.models import PyTesseractModel


class ModelFactory:
    _services = {
        InferenceType.DETECTION: {
            ModelProvider.ROBOFLOW: RoboflowDetectionModel
        },
        InferenceType.CLASSIFICATION: {
            ModelProvider.ROBOFLOW: RoboflowClassificationModel
        },
        InferenceType.OCR: {
            ModelProvider.PYTESSERACT: PyTesseractModel
        }
    }

    @classmethod
    def create(cls, model: ModelConfig) -> Union[DetectionModelBase | OcrModelBase | ClassificationModelBase]:
        service_cls = cls._services[model.inference][model.provider]
        return service_cls(**model.config)