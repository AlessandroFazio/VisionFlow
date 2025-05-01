from collections import defaultdict
from typing import Dict, List, Optional, Callable

import numpy as np
from prefect import task
from visionflow.core.inference.detection.base import DetectionResult
from visionflow.core.pipeline.base import Exchange, StepBase


class RouteDetectionStep(StepBase):
    def __init__(
        self,
        routes: Dict[str, StepBase],
        in_key: str,
        out_key: Optional[str] = None
    ) -> None:
        self.routes = routes
        super().__init__(name="route_detection", in_key=in_key, out_key=out_key or in_key)

    def _group_by_class(self, detections: List[DetectionResult]) -> Dict[str, List[DetectionResult]]:
        grouped: Dict[str, List[DetectionResult]] = defaultdict(list)
        for d in detections:
            grouped[d.class_name].append(d)
        return grouped

    def _crop(self, det: DetectionResult, img: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = det.xyxy
        return img[y1:y2, x1:x2]

    def _dispatch(
        self,
        exchange: Exchange,
        executors: Dict[str, Callable[[Exchange], None]]
    ) -> None:
        for det in exchange.detections[self.in_key]:
            executor = executors.get(det.class_name)
            if not executor:
                continue

            copy = exchange.copy()
            copy.detections[self.out_key] = [det]
            executor(copy)

    def process(self, exchange: Exchange) -> Exchange:
        python_routes = {n: route.process for n, route in self.routes.items()}
        self._dispatch(exchange, python_routes)
        return exchange

    def to_prefect(self) -> task:
        prefect_routes = {n: step.to_prefect() for n, step in self.routes.items()}

        @task(name=self.name)
        def step_task(exchange: Exchange) -> Exchange:
            self._dispatch(exchange, prefect_routes)
            return exchange

        return step_task