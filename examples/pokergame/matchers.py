from visionflow.core.regex.base import RegexMatcherBase
from visionflow.core.regex.builders import RegexMatcherBuilders


class Patterns:
    CHIPS_AMOUNT = "^(?P<currency>[€$])?\\s*(?P<amount>\\d+[.,]?\\d*)$"
    PLAYER_ACTION = "^(?P<action>(Check|Fold|Raise|Call|Rilancia|Chiama))$"
    PLAYER_STATE = "^(?P<state>(All-in|In Sit-out))$"
    PLAYER_USERNAME = "^(?P<username>[a-zA-Z0-9_]+)$"
    SEAT_STATE = "^(?P<state>(Empty|Seat|Posto|Vuoto))$"


class Matchers:
    @staticmethod
    def chips_amount() -> RegexMatcherBase:
        return (
            RegexMatcherBuilders.by_line_matcher()
                .rule(pattern=Patterns.CHIPS_AMOUNT, first_match=True)
                .build()
        )

    @staticmethod
    def seat_info() -> RegexMatcherBase:
        return (
            RegexMatcherBuilders.by_line_matcher()
                .rule(id="username", pattern=Patterns.PLAYER_USERNAME, first_match=True, priority=10)
                .rule(id="action", pattern=Patterns.PLAYER_ACTION, first_match=True, priority=10)
                .rule(id="state", pattern=Patterns.PLAYER_STATE, first_match=True, priority=10)
                .rule(id="stack", pattern=Patterns.CHIPS_AMOUNT, first_match=True, priority=10)
                .rule(id="seat_state", pattern=Patterns.SEAT_STATE, first_match=True, priority=10)
                .build()
        )