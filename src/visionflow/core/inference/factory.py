from visionflow.core.inference.classification.services import RoboflowClassificationService
from visionflow.core.inference.config import ModelConfig, ProviderType, ServiceType
from visionflow.core.inference.detection.base import DetectionServiceBase
from visionflow.core.inference.detection.services import RoboflowDetectionService
from visionflow.core.inference.ocr.services import LocalTesseractService


class InferenceServiceFactory:
    _services = {
        ServiceType.DETECTION: {
            ProviderType.ROBOFLOW: RoboflowDetectionService
        },
        ServiceType.CLASSIFICATION: {
            ProviderType.ROBOFLOW: RoboflowClassificationService
        },
        ServiceType.OCR: {
            ProviderType.LOCAL_TESSERACT: LocalTesseractService
        }
    }

    @classmethod
    def create(cls, model: ModelConfig) -> DetectionServiceBase:
        service_cls = cls._services[model.type][model.provider]
        return service_cls(**model.config)