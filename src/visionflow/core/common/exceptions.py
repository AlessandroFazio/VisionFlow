from abc import ABC, abstractmethod


class VisionFlowException(ABC, Exception):
    @abstractmethod
    def get_cause(self) -> str:
        pass


class ContextAccessException(VisionFlowException):
    def __init__(self, message: str) -> None:
        self.message = message

    def get_cause(self) -> str:
        return self.message