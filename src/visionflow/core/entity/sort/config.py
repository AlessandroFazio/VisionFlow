from enum import Enum

from visionflow.typing.enums import EnumValueLookupMixin


class SortType(EnumValueLookupMixin):
    LEX = "lex"
    CLOCKWISE = "clockwise"


class SortKey(EnumValueLookupMixin):
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    X_AVG = "x_avg"
    Y_AVG = "y_avg"