# visionflow/regex/builders.py

from typing import List, Optional, Set, Tuple
from typing_extensions import Self
from visionflow.core.regex.base import RegexMatcherRule
from visionflow.core.regex.matchers import RegexMatcherByLine, RegexMatcherFullText


class RegexMatcherBuilderBase:
    def __init__(self):
        self._rules: List[RegexMatcherRule] = []
        self._rule_ids: Set[str] = set()

    def rule(
        self, 
        pattern: str, 
        priority: int=10, 
        id: str="default",
        target_linenos: Optional[List[int]]=None, 
        first_match: bool = False,
        case_insensitive: bool = False
    ) -> "Self":
        if id in self._rule_ids:
            raise ValueError("")
        self._rule_ids.add(id)
        self._rules.append(RegexMatcherRule(
            id=id,
            priority=priority,
            pattern=pattern,
            target_linenos=target_linenos,
            first_match=first_match,
            case_insensitive=case_insensitive
        ))
        return self


class ByLineRegexMatcherBuilder(RegexMatcherBuilderBase):
    def __init__(self):
        super().__init__()
        self._global_exclusive: List[Tuple[str, ...]] = []
        self._line_exclusive: List[Tuple[str, ...]] = []

    def line_exclusive(self, *group: str) -> "ByLineRegexMatcherBuilder":
        self._line_exclusive.append(group)
        return self

    def global_exclusive(self, *group: str) -> "ByLineRegexMatcherBuilder":
        self._global_exclusive.append(group)
        return self

    def build(self) -> RegexMatcherByLine:
        return RegexMatcherByLine(self._rules, self._global_exclusive, self._line_exclusive)


class FullTextRegexMatcherBuilder(RegexMatcherBuilderBase):
    def __init__(self):
        super().__init__()
        self._mutually_exclusive: List[Tuple[str, ...]] = []

    def mutually_exclusive(self, *group: str) -> "FullTextRegexMatcherBuilder":
        self._mutually_exclusive.append(group)
        return self

    def build(self) -> RegexMatcherFullText:
        return RegexMatcherFullText(self._rules, self._mutually_exclusive)


class RegexMatcherBuilders:
    @staticmethod
    def by_line_matcher() -> ByLineRegexMatcherBuilder:
        return ByLineRegexMatcherBuilder()

    @staticmethod
    def full_text_matcher() -> FullTextRegexMatcherBuilder:
        return FullTextRegexMatcherBuilder()
