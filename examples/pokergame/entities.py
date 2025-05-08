from typing import List
from visionflow.core.entity.base import EntityBase
from visionflow.core.entity.reflection.converter.defaults import EntityRefIsPresent
from visionflow.core.entity.reflection.decorator.entity import entity
from visionflow.core.entity.reflection.descriptor.fields import ClassificationLabel, EntityRef, OcrRegex

from .converters import CashAmountConverter, SeatStateConverter


@entity(detection_class="card")
class Card:
    value: str = ClassificationLabel(allowed_labels=[
      str(r) + s 
      for r in [2, 3, 4, 5, 6, 7, 8, 9, 'T', 'J', 'Q', 'K', 'A'] 
      for s in ['h', 's', 'c', 'd']
    ])

@entity(detection_class="dealer_button")
class DealerButton:
  pass

@entity(detection_class="player_in")
class PlayerIn:
  pass

@entity(detection_class="chips")
class ChipsAmount:
  value: float = OcrRegex(converter=CashAmountConverter())

@entity(detection_class="player_info", parent_selector="exact")
class Player:
  username: str = OcrRegex(rule_selector="player_username", match_key="player_username")
  action: str = OcrRegex(rule_selector="player_action", match_key="player_action")
  state: str = OcrRegex(rule_selector="player_state", match_key="player_state")
  stack: float = OcrRegex(rule_selector="player_stack", match_key="player_stack", converter=CashAmountConverter())
  bet: ChipsAmount
  is_dealer: bool = EntityRef(entity_cls=DealerButton, converter=EntityRefIsPresent())

@entity(detection_class="player_info")
class TableSeat:
    empty: bool = OcrRegex(match_key="seat_state", rule_selector="seat_state", converter=SeatStateConverter())
    player: Player

@entity(fixed_point=(0.5, 0.5))
class Community:
    cards: List[Card] = EntityRef(sort_type="lex", sort_keys=["x_min"])
    pots: List[ChipsAmount]

@entity()
class PokerTable(EntityBase):
    seats: List[TableSeat]
    community: Community