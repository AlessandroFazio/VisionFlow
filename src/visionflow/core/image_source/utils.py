import cv2
import numpy as np


def image_from_bytes(b: bytes) -> np.ndarray:
    img = np.frombuffer(b, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img