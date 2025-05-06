from visionflow.core.pipeline.builder.groups import FilterCrop, FilterCropClassify, FilterMask, OcrRegexMatcher, StepGroup
from visionflow.core.pipeline.builder.pipeline import PipelineBuilder

from .entities import Card, ChipsAmount, DealerButton, PlayerIn, TableSeat
from .matchers import Matchers


class CardRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder: 
        return (
            builder
              .apply(FilterCropClassify("poker/card_classifier", min_conf=0.70))
              .build_entity(Card)
        )


class SeatInfoRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .apply(FilterCrop(min_conf=0.70))
            .apply(OcrRegexMatcher("poker/player_info_ocr", Matchers.seat_info()))
            .build_entity(TableSeat)
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
              .split_by_detections()
                
                .for_class("chips")
                  .apply(FilterMask(min_conf=0.70))
                  .apply(OcrRegexMatcher("poker/chips_ocr", Matchers.chips_amount()))
                  .build_entity(ChipsAmount)
                .end_class()
              
              .end_split()
      )