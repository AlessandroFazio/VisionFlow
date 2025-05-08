from visionflow.core.regex_matcher.base import RegexMatcherBase
from visionflow.core.regex_matcher.builders import RegexMatchers


class Patterns:
    CASH_AMOUNT = "^(?P<currency>[€$])?\\s*(?P<amount>\\d+[.,]?\\d*)$"
    PLAYER_ACTION = "^(?P<action>(Check|Fold|Raise|Call|Rilancia|Chiama))$"
    PLAYER_STATE = "^(?P<state>(All-in|In Sit-out))$"
    PLAYER_USERNAME = "^(?P<username>[a-zA-Z0-9_]+)$"
    SEAT_STATE = "^(?P<state>(Empty|Seat|Posto|Vuoto))$"


class Matchers:
    @staticmethod
    def chips_amount() -> RegexMatcherBase:
        return (
            RegexMatchers.full_text()
                .rule(pattern=Patterns.CASH_AMOUNT, first_match=True)
                .build()
        )

    @staticmethod
    def seat_info() -> RegexMatcherBase:
        return (
            RegexMatchers.line_chunks()
                .rule(id="player_username", pattern=Patterns.PLAYER_USERNAME, target_linenos=[0], first_match=True, priority=10)
                .rule(id="player_action", pattern=Patterns.PLAYER_ACTION, target_linenos=[0], first_match=True, priority=5)
                .rule(id="player_state", pattern=Patterns.PLAYER_STATE, target_linenos=[1], first_match=True, priority=5)
                .rule(id="player_stack", pattern=Patterns.CASH_AMOUNT, target_linenos=[1], first_match=True, priority=10)
                .rule(id="seat_state", pattern=Patterns.SEAT_STATE, priority=0)
                .mutually_exclusive("player_username", "player_action")
                .mutually_exclusive("player_state", "player_stack")
                .mutually_exclusive("seat_state", ("player_username", "player_action", "player_state", "player_stack"))
                .build()
        )