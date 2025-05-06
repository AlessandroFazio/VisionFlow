# visionflow/regex/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(order=True)
class RegexMatcherRule:
    id: str = field(compare=False)
    priority: int = field(compare=True)
    pattern: str = field(compare=False)
    target_linenos: List[int] = field(default_factory=list, compare=False)
    first_match: bool = False
    case_insensitive: bool = False
    _compiled: re.Pattern = field(init=False, compare=False)
    _re_flags: int = field(init=False, compare=False, default=0)

    def __post_init__(self):
        if self.case_insensitive:
            self._re_flags |= re.IGNORECASE
        self._compiled = re.compile(self.pattern, self._re_flags)

    def match(self, text: str) -> Iterable[re.Match[str]]:
        if self.first_match:
            match = self._compiled.match(text)
            return iter([match] if match else [])
        return self._compiled.finditer(text)


@dataclass
class RegexMatchResult:
    rule_id: str
    text: str
    matches: Dict[str, str]
    lineno: Optional[int] = None


class RegexMatcherBase(ABC):
    def __init__(self, rules: List[RegexMatcherRule]):
        self.rules = sorted(rules, reverse=True)

    @abstractmethod
    def match(self, text: str) -> List[RegexMatchResult]:
        pass
