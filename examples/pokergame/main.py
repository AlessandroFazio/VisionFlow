import PIL
import PIL.Image

from examples.pokergame.models import Models
from visionflow.core.visionflow import VisionFlow

from .entities import PokerTable
from .recognizers import CardRecognizer, ChipsAmountRecognizer, DealerButtonRecognizer, PlayerInRecognizer, SeatInfoRecognizer


def main():    
    img_path = ""
    img_bytes = PIL.Image.open(img_path).tobytes()

    pipeline = (
        VisionFlow.pipeline("pokerstars_recognizer")
          .detect(Models.table_detection())
          .split_by_detections()
            
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