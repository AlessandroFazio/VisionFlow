from typing import List

from babel.numbers import parse_decimal

from visionflow.core.entity.reflection.converter.base import RegexMatcherConverterBase
from visionflow.core.regex_matcher.base import RegexMatchResult


class CashAmountConverter(RegexMatcherConverterBase[float]):
    def convert(self, results: List[RegexMatchResult]) -> float:
        matches = results[0].matches
        raw_amount = matches.get("amount")
        if not raw_amount:
            return None
        
        currency = matches.get("currency")
        locale = "it_IT" if currency == "€" else "en_US"
        try:
            return float(parse_decimal(raw_amount), locale=locale)
        except Exception:
            return None


class SeatStateConverter(RegexMatcherConverterBase[bool]):
    def convert(self, results: List[RegexMatchResult]) -> bool:
        if len(results) != 2:
            return False
        text = " ".join(r.matches.get("state", "") for r in results)
        return text.strip() in ["Posto Vuoto", "Empty Seat"]