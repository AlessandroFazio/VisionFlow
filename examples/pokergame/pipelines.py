from visionflow.core.pipeline.pipeline import Pipeline
from visionflow.core.visionflow import VisionFlow

from examples.pokergame.entities import PokerTable
from examples.pokergame.models import Models
from examples.pokergame.recognizers import CardRecognizer, ChipsAmountRecognizer, DealerButtonRecognizer, PlayerInRecognizer, SeatInfoRecognizer


class Pipelines:
    @staticmethod
    def poker_table() -> Pipeline:
        return (
            VisionFlow.pipeline("pokertable_recognizer")
                .detect(Models.table_detection())
                .branch_detections()
                    
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
                    
                .end_branch()        
                .build_entities(PokerTable)
                .resolve_entities()
                
                .build()
        )