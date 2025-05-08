import numpy as np

from visionflow.core.inference.classification.base import ClassificationResult, ClassificationModelBase
from visionflow.core.inference.mixins.roboflow import RoboflowModelMixin


class RoboflowClassificationModel(RoboflowModelMixin, ClassificationModelBase):

    def classify(self, img: np.ndarray) -> ClassificationResult:
        result = self._model.infer(img)[0]
        result = result.predictions[0]
        result = ClassificationResult(
            class_name=str(result.class_name), 
            confidence=float(result.confidence)
        )
        return result