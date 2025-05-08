from abc import ABC

from visionflow.typing.enums import EnumValueLookupMixin


class ModelProvider(EnumValueLookupMixin):
    ROBOFLOW = "roboflow"
    PYTESSERACT = "pytesseract"


class InferenceType(EnumValueLookupMixin):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    OCR = "ocr"

class InferenceModelBase(ABC):
    pass