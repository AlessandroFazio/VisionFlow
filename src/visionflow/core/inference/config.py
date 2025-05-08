from typing import Optional, Union

from pydantic import BaseModel

from visionflow.core.inference.base import InferenceType, ModelProvider


class RoboflowModelConfig(BaseModel):
    model_id: str


class TesseractModelConfig(BaseModel):
    psm: int
    chars_whitelist: Optional[str] = ""
    lang: Optional[str] = "eng+osd"


class ModelConfig(BaseModel):
    name: str
    inference: InferenceType
    provider: ModelProvider
    config: Union[RoboflowModelConfig | TesseractModelConfig]