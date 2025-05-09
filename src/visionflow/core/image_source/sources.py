# from_bytes.py
from typing import Iterator, List

import cv2
import numpy as np
import boto3

from visionflow.core.image_source.base import ImageSourceBase
from visionflow.core.image_source.utils import image_from_bytes


class BytesImageSource(ImageSourceBase):
    def __init__(self, bytes_list: List[bytes]) -> None:
        self.bytes_list = bytes_list

    def images(self) -> Iterator[np.ndarray]:
        for b in self.bytes_list:
            yield image_from_bytes(b)


class FileImageSource(ImageSourceBase):
    def __init__(self, paths: List[str]) -> None:
        self.paths = paths

    def images(self) -> Iterator[np.ndarray]:
        for path in self.paths:
            yield cv2.imread(path, cv2.IMREAD_COLOR)


class S3ImageSource(ImageSourceBase):
    def __init__(self, bucket: str, keys: List[str]) -> None:
        self._bucket = bucket
        self._keys = keys
        self.s3 = boto3.client("s3")

    def images(self) -> Iterator[np.ndarray]:
        for key in self._keys:
            obj = self.s3.get_object(Bucket=self._bucket, Key=key)
            yield image_from_bytes(obj["Body"].read())
