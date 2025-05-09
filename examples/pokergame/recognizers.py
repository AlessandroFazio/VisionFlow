from examples.pokergame.models import Models
from visionflow.core.pipeline.builder.groups import FilterCrop, FilterCropClassify, FilterMask, OcrRegexMatcher, StepGroup
from visionflow.core.pipeline.builder.pipeline import PipelineBuilder

from .entities import Card, ChipsAmount, DealerButton, PlayerIn, TableSeat
from .matchers import Matchers


class CardRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder: 
        return (
            builder
              .apply(FilterCropClassify(Models.card_classifier(), min_conf=0.70))
              .build_entities(Card)
        )


class SeatInfoRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .apply(FilterCrop(min_conf=0.70))
            .resize((2,2))
            .binarize()
            .apply(OcrRegexMatcher(Models.player_info_ocr(), Matchers.seat_info()))
            .build_entities(TableSeat)
      )


class PlayerInRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .filter(min_conf=0.70)
            .build_entities(PlayerIn)
      )

 
class DealerButtonRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .filter(min_conf=0.70)
            .build_entities(DealerButton)
      )


class ChipsAmountRecognizer(StepGroup):
    def apply(self, builder: PipelineBuilder) -> PipelineBuilder:
      return (
          builder
            .apply(FilterCrop(min_conf=0.70))
              .detect(Models.chips_amount_detection())
              .branch_detections()
                
                .for_class("chips")
                  .apply(FilterMask(min_conf=0.70))
                  .resize((2,2))
                  .binarize()
                  .apply(OcrRegexMatcher(Models.chips_amount_ocr(), Matchers.chips_amount()))
                  .build_entities(ChipsAmount)
                .end_class()
              
              .end_branch()
      )