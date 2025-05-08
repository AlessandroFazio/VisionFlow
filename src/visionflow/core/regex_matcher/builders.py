# visionflow/regex/builders.py

from itertools import chain
from typing import Any, Iterable, List, Optional, Set, Tuple, Union
from visionflow.core.regex_matcher.base import RegexMatcherBase, RegexMatcherMode, RegexMatcherRule
from visionflow.core.regex_matcher.matchers import FullTextRegexMatcher, LineChunksRegexMatcher


class RegexMatcherBuilder:
    _types_map = {
        RegexMatcherMode.LINE_CHUNKS: LineChunksRegexMatcher,
        RegexMatcherMode.FULL_TEXT: FullTextRegexMatcher
    }
    
    def __init__(self, mode: RegexMatcherMode) -> None:
        self._mode: str = mode
        self._rules: List[RegexMatcherRule] = []
        self._rule_ids: Set[str] = set()
        self._mutually_exclusive_groups: List[Any] = []

    def rule(
        self, 
        pattern: str, 
        priority: int=10, 
        id: str="default",
        target_linenos: Optional[List[int]]=None, 
        first_match: bool = False,
        case_insensitive: bool = False
    ) -> "RegexMatcherBuilder":
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
    
    def _expand_group_pairs(items: Iterable[Union[str, Tuple[str, ...]]]) -> Set[Tuple[str, str]]:
        groups = [(item,) if isinstance(item, str) else item for item in items]
        pairs = set()
        for i, group in enumerate(groups):
            other_items = chain.from_iterable(groups[:i] + groups[i+1:])
            for item1 in group:
                for item2 in other_items:
                    if item1 == item2:
                        raise ValueError("")
                    pairs.add(tuple(sorted((item1, item2))))
        return pairs
    
    def mutually_exclusive(self, *group: Union[str | Tuple[str, ...]]) -> "RegexMatcherBuilder":
        if len(group) < 2:
            raise ValueError("")
        self._mutually_exclusive_groups.extend(
            self._expand_group_pairs(group)
        )
        return self

    def build(self) -> "RegexMatcherBase":
        matcher_type = self._types_map.get(self._mode)
        if not matcher_type:
            raise ValueError("")
        return matcher_type(
            rules=self._rules, 
            mutual_exclusion=self._mutually_exclusive_groups
        )
        

class RegexMatchers:
    @staticmethod
    def line_chunks() -> RegexMatcherBuilder:
        return RegexMatcherBuilder(RegexMatcherMode.LINE_CHUNKS)
    
    @staticmethod
    def full_text() -> RegexMatcherBuilder:
        return RegexMatcherBuilder(RegexMatcherMode.FULL_TEXT)