from prefect import flow

from examples.pokergame.pipelines import Pipelines
from visionflow.core.image_source.factory import ImageSourceFactory
from visionflow.core.visionflow import VisionFlow

from .entities import PokerTable


@flow(name="example")
def pokertable_pipeline(config: dict):
    backend = VisionFlow.prefect_backend()
    
    source = ImageSourceFactory.from_dict(config)
    contexts = [
      (
        VisionFlow.run_context()
          .with_image(image)
          .with_entities(PokerTable)
          .with_backend(backend)
      ) for image in source.images()
    ]

    backend.run_batch(Pipelines.poker_table(), contexts)


if __name__ == "__main__":
    pokertable_pipeline.deploy(...)