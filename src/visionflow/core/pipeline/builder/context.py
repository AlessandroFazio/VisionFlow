# visionflow/core/pipeline/context_builder.py

from typing import Type, List, Optional
import numpy as np

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.registry.registries import GraphEntityRegistry
from visionflow.core.pipeline.base import Exchange, StepRunContext
from visionflow.core.pipeline.backend.base import BackendBase, StepDispatcherBase


class RunContextBuilder:
    def __init__(self) -> None:
        self._entity_classes: List[Type[EntityBase]] = []
        self._image: Optional[np.ndarray] = None
        self._dispatcher: Optional[StepDispatcherBase] = None

    def with_entities(self, *classes: Type[EntityBase]) -> "RunContextBuilder":
        self._entity_classes.extend(classes)
        return self

    def with_image(self, img: np.ndarray) -> "RunContextBuilder":
        self._image = img
        return self

    def with_backend(self, backend: BackendBase) -> "RunContextBuilder":
        self._dispatcher = backend.dispatcher()
        return self

    def build(self) -> StepRunContext:
        if self._image is None:
            raise ValueError("Image must be provided")
        if self._dispatcher is None:
            raise ValueError("Dispatcher must be provided")

        entity_registry = GraphEntityRegistry()
        for cls in self._entity_classes:
            entity_registry.register_class(cls)

        context = StepRunContext(
            exchange=Exchange.from_image(self._image),
            entity_registry=entity_registry,
            dispatcher=self._dispatcher
        )

        return context