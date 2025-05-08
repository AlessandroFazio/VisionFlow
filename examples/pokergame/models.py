from visionflow.core.inference.builders import ModelBuilder
from visionflow.core.inference.classification.base import ClassificationModelBase
from visionflow.core.inference.config import RoboflowModelConfig, TesseractModelConfig
from visionflow.core.inference.detection.base import DetectionModelBase
from visionflow.core.inference.ocr.base import OcrModelBase


class Models:
    @staticmethod
    def table_detection() -> DetectionModelBase:
        return (
            ModelBuilder()
                .inference("detection")
                .provider("roboflow")
                .config(RoboflowModelConfig(model_id=""))
        )
        
    @staticmethod
    def card_classifier() -> ClassificationModelBase:
        return (
            ModelBuilder()
                .inference("classification")
                .provider("roboflow")
                .config(RoboflowModelConfig(model_id=""))
        )
        
    @staticmethod
    def chips_amount_detection() -> DetectionModelBase:
        return (
            ModelBuilder()
                .inference("detection")
                .provider("roboflow")
                .config(RoboflowModelConfig(model_id=""))
        )
        
    @staticmethod
    def chips_amount_ocr() -> OcrModelBase:
        return (
            ModelBuilder()
                .inference("ocr")
                .provider("pytesseract")
                .config(TesseractModelConfig(
                    psm=7,
                    chars_whitelist="",
                    lang=""
                ))
        )
        
    @staticmethod
    def player_info_ocr() -> OcrModelBase:
        return (
            ModelBuilder()
                .inference("ocr")
                .provider("pytesseract")
                .config(TesseractModelConfig(
                    psm=7,
                    chars_whitelist="",
                    lang=""
                ))
        )