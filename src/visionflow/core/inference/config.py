from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel


class ProviderType(Enum):
    ROBOFLOW = "roboflow"
    LOCAL_TESSERACT = "tesseract/local"


class ServiceType(Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    OCR = "ocr"


class RoboflowModelConfig(BaseModel):
    model_id: str


class TesseractModelConfig(BaseModel):
    psm: int
    chars_whitelist: Optional[str] = ""
    lang: Optional[str] = "eng+osd"


class ModelConfig(BaseModel):
    name: str
    type: ServiceType
    provider: ProviderType
    config: Union[RoboflowModelConfig | TesseractModelConfig]