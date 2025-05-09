# visionflow/core/image_source/config.py

from pydantic import BaseModel
from typing import List, Literal, Union


class FileImageSourceConfig(BaseModel):
    type: Literal["file"]
    paths: List[str]


class S3ImageSourceConfig(BaseModel):
    type: Literal["s3"]
    bucket: str
    keys: List[str]


class BytesImageSourceConfig(BaseModel):
    type: Literal["bytes"]
    base64_images: List[str]


ImageSourceConfig = Union[FileImageSourceConfig, S3ImageSourceConfig, BytesImageSourceConfig]
