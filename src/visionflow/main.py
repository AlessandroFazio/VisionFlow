from typing import Callable, Type
import PIL
import PIL.Image
import yaml

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.decorators import entity
from visionflow.core.entity.reflection.descriptors import ClassificationLabel, EntityRef, OcrRegex
from visionflow.core.pipeline.builder.groups import FilterCrop, FilterCropClassify, FilterCropOcr, FilterMaskOcr, StepGroup
from visionflow.core.pipeline.builder.pipeline import PipelineBuilder
from visionflow.core.visionflow import VisionFlow, VisionFlowConfig

@entity()
class Card(EntityBase):
    value: str = ClassificationLabel(allowed_labels=[])

@entity()
class SeatState(EntityBase):
    empty: bool = OcrRegex(pattern="", converter=...)

@entity()
class PokerPlayer(EntityBase):
    username: str = OcrRegex(pattern)

@entity()
class TableSeat(EntityBase):
    state: SeatState = EntityRef()
    player: PokerPlayer = EntityRef()
    is_dealer: bool = EntityRef()

@entity()
class PokerTable(EntityBase):
    state: SeatState = EntityRef()
    player: PokerPlayer = EntityRef()
    is_dealer: bool = EntityRef()


class CardRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder: 
        return (
            builder
              .apply(FilterCropClassify("poker/card_classifier", min_conf=0.70))
              .build_entity(Card)
        )
    
class PlayerInfoRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .apply(FilterCropOcr("poker/player_info_ocr", min_conf=0.70, reshape=(2,2)))
            .build_entity(SeatState)
      )
    
class PlayerInRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .filter(min_conf=0.70)
            .build_entity(PlayerIn)
      )
    
class DealerButtonRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .filter(min_conf=0.70)
            .build_entity(DealerButton)
      )

class ChipsAmountRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .apply(FilterCrop(min_conf=0.70))
              .detect("poker/chips_detector")
              .split_detections()
                .for_class("chips")
                  .apply(FilterMaskOcr("poker/chips_ocr", min_conf=0.70, reshape=(2,2)))
                  .build_entity(ChipsAmount)
                .end_class()
              .end_split()
      )


def main():
    config_path = ""
    with open(config_path) as f:
        content = yaml.safe_load(f)
    
    img_path = ""
    img_bytes = PIL.Image.open(img_path).tobytes()
    
    config = VisionFlowConfig.model_validate(content)
    vf = VisionFlow(config)

    pipeline = (
        vf.builder("pokerstars_recognizer")
          .detect("poker/table_detection")
          .split_detections()
            .for_class("card")
              .apply(CardRecognizer())
            .end_class()
            .for_class("player_info")
              .apply(PlayerInfoRecognizer())
            .end_class()
            .for_class("player_in")
              .apply(PlayerInRecognizer())
            .end_class()
            .for_class("dealer_button")
              .apply(DealerButtonRecognizer())
            .end_class()
            .for_class("chips_amount")
              .apply(ChipsAmountRecognizer())
            .end_class()
          .end_split()
          .build_entity(PokerTable)
          .build()
      )
    
    pipeline.run(img_bytes)


if __name__ == "__main__":
    main()