import PIL
import PIL.Image
import yaml

from visionflow.core.visionflow import VisionFlow, VisionFlowConfig

from .entities import PokerTable
from .recognizers import CardRecognizer, ChipsAmountRecognizer, DealerButtonRecognizer, PlayerInRecognizer, SeatInfoRecognizer


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
              .apply(SeatInfoRecognizer())
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