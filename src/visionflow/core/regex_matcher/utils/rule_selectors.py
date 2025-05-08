from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from visionflow.core.regex_matcher.base import RegexMatcherRule
from visionflow.core.regex_matcher.utils.state import RuleExecutionState


class RuleSelectorBase(ABC):
    @abstractmethod
    def select(self, chunk_id: Optional[int], state: RuleExecutionState) -> Iterable[RegexMatcherRule]:
        pass

        
class LineChunksRuleSelector(RuleSelectorBase):
    def __init__(self, rules: List[RegexMatcherRule]):
        self._index: Dict[int, List[RegexMatcherRule]] = defaultdict(list)
        for r in rules:
            for ln in r.target_linenos or [-1]:
                self._index[ln].append(r)

    def select(self, chunk_id: Optional[int], state: RuleExecutionState) -> Iterable[RegexMatcherRule]:
        rules = self._index.get(chunk_id, []) + self._index.get(-1, [])
        return sorted((r for r in rules if state.is_enabled(r.id)), key=lambda r: r.priority)


class FullTextRuleSelector(RuleSelectorBase):
    def __init__(self, rules: List[RegexMatcherRule]):
        self._rules = rules

    def select(self, _: Optional[int], state: RuleExecutionState) -> Iterable[RegexMatcherRule]:
        return sorted((r for r in self._rules if state.is_enabled(r.id)), key=lambda r: r.priority)