from typing import List

from babel.numbers import parse_decimal

from visionflow.core.entity.reflection.types import OcrRegexConverter, RegexGroupDictType


class Converters:
  @staticmethod
  def chips_amount(re_gd_list: List[RegexGroupDictType]) -> float:
    if not re_gd_list:
      return None
    gd = re_gd_list[0]
    locale = (
      "it_IT" if str(gd.get("currency", "")) == "€" 
      else "en_US"
    )
    try:
      return float(parse_decimal(str(gd.get("amount", "")), locale=locale))
    except Exception:
      return None

  @staticmethod
  def ocr_regex(re_gd_list: List[RegexGroupDictType], key: str) -> OcrRegexConverter:
    return (
      re_gd_list[0].get(key) 
      if re_gd_list else None
    )

  @staticmethod
  def seat_state(re_gd_list: List[RegexGroupDictType]) -> OcrRegexConverter:
    if not re_gd_list:
      return None
    states = [s for s in re_gd_list[0].get("state")]
    return (
      re_gd_list[0].get("state")
    )