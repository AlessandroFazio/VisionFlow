# visionflow/core/image_source/factory.py

from pydantic import TypeAdapter
from visionflow.core.image_source.base import ImageSourceBase
from visionflow.core.image_source.config import ImageSourceConfig
import base64

from visionflow.core.image_source.sources import BytesImageSource, FileImageSource, S3ImageSource


class ImageSourceFactory:
    @classmethod
    def create(cls, config: ImageSourceConfig) -> ImageSourceBase:
        cfg_type = config.type
        
        if cfg_type == "file":
            return FileImageSource(config.paths)
    
        if cfg_type == "s3":
            s3_paths = [(config.bucket, key) for key in config.keys]
            return S3ImageSource(s3_paths)
    
        if cfg_type == "bytes":
            decoded = [base64.b64decode(b) for b in config.base64_images]
            return BytesImageSource(decoded)
    
        raise ValueError(f"Unsupported image source type: {cfg_type}")
    
    @classmethod
    def from_dict(cls, config: dict) -> ImageSourceBase:
        config = TypeAdapter(ImageSourceConfig).validate_python(config)
        return cls.create(config)
