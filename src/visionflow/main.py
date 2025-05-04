from typing import Callable, Type
import PIL
import PIL.Image
import yaml

from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.descriptors import ClassificationLabelField, EntityRefField, OcrRegexField
from visionflow.core.visionflow import VisionFlow, VisionFlowConfig

def branch_binding(name: str) -> Callable[[Type[EntityBase]], Type[EntityBase]]:
    def decorator(cls: Type[EntityBase]) -> Type[EntityBase]:
        cls.__vf_meta__.branch_binding = name
        return cls
    return decorator


class Card(EntityBase):
    value: str = ClassificationLabelField(allowed_labels=[])

@branch_binding(name="player_info")
class SeatState(EntityBase):
    empty: bool = OcrRegexField(pattern="", converter=...)

@branch_binding(branch="player_info")
class PokerPlayer(EntityBase):
    username: str = OcrRegexField(pattern)

class TableSeat(EntityBase):
    state: SeatState = EntityRefField()
    player: PokerPlayer = EntityRefField()
    is_dealer: bool = EntityRefField()


def main():
    config_path = ""
    with open(config_path) as f:
        content = yaml.safe_load(f)
    
    img_path = ""
    img_bytes = PIL.Image.open(img_path).tobytes()
    
    config = VisionFlowConfig.model_validate(content)
    vf = VisionFlow(config)
    builder = vf.builder("pokerstars_recognizer")

    pipeline = (
        builder
          .detect("poker/table_detection")
          .split_by_detections()
            
            .branch("card")
              .filter(min_conf=0.70)
              .crop_to_detection()
              .classify("poker/card_classifier")
            .end_branch()
            
            .branch("player_info")
              .filter(min_conf=0.70)
              .crop_to_detection()
              .resize((2,2))
              .binarize()
              .ocr("poker/player_info_ocr")
            .end_branch()
            
            .branch("player_in")
              .filter(min_conf=0.70)
            .end_branch()
            
            .branch("dealer_button")
              .filter(min_conf=0.70)
            .end_branch()
            
            .branch("chips_amount")
              .filter(min_conf=0.70)
              .crop_to_detection()
              .detect("poker/chips_detector")
              .split_by_detections()
                .branch("chips")
                  .filter(min_conf=0.70)
                  .mask_detection()
                  .resize((2,2))
                  .binarize()
                  .ocr("")
                .end_branch()
              .end_split()
            .end_branch()
          
          .end_split()
          .root_entity(Card)
          .build()
      )
    
    pipeline.run(img_bytes)


if __name__ == "__main__":
    main()