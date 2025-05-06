from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.decorators import entity
from visionflow.core.entity.reflection.descriptors import ClassificationLabel, EntityRef, OcrRegex

from .converters import Converters


@entity(detection_class="card")
class Card:
    value: str = ClassificationLabel(converter=lambda labels: labels[0])


@entity(detection_class="chips")
class ChipsAmount:
  value: float = OcrRegex(converter=Converters.chips_amount)


@entity(detection_class="player_info", parent_selector="exact")
class Player:
  username: str = OcrRegex(converter=lambda x: Converters.ocr_regex(x, "username"))
  action: str = OcrRegex(converter=lambda x: Converters.ocr_regex(x, "action"))
  state: str = OcrRegex(converter=lambda x: Converters.ocr_regex(x, "state"))
  bet: float = EntityRef()
  

@entity(detection_class="dealer_button")
class DealerButton:
  pass


@entity(detection_class="player_in")
class PlayerIn:
  pass


@entity()
class TableSeat:
    empty: bool = OcrRegex(converter=lambda x: Converters.ocr_regex(x, "state"))
    player: Player = EntityRef()
    is_dealer: bool = EntityRef(entity_cls=DealerButton, converter=lambda x: x is not None)


@entity()
class PokerTable(EntityBase):
    state: SeatState = EntityRef()
    player: PokerPlayer = EntityRef()
    is_dealer: bool = EntityRef()