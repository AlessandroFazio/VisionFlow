import os
import inference
from inference.core.models.base import Model
get_model = getattr(inference, "get_model")


class RoboflowModelMixin:
    _API_KEY_ENV = "ROBOFLOW_API_KEY"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.api_key = os.environ.get(self._API_KEY_ENV)
        if not self.api_key:
            raise ValueError("")
        self._model: Model = get_model(model_id=model_id, api_key=self.api_key)