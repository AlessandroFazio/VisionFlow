from collections import defaultdict
from typing import Dict, List, Set, Tuple
from visionflow.core.regex_matcher.base import RegexMatchResult, RegexMatcherBase, RegexMatcherRule
from visionflow.core.regex_matcher.utils.chunkers import ChunkerStrategyBase, FullTextChunker, LineChunker
from visionflow.core.regex_matcher.utils.rule_selectors import FullTextRuleSelector, LineChunksRuleSelector, RuleSelectorBase
from visionflow.core.regex_matcher.utils.state import RuleExecutionState


class RegexMatcherTemplate(RegexMatcherBase):
    def __init__(
        self, 
        rules: List[RegexMatcherRule], 
        mutual_exclusion: Set[Tuple[str, str]], 
        stop_on_first_match: bool,
        rule_selector: RuleSelectorBase,
        chunker: ChunkerStrategyBase
    ) -> None:
        super().__init__(rules=rules, mutual_exclusion=mutual_exclusion)
        self.stop_on_first_match = stop_on_first_match
        self.rule_selector = rule_selector 
        self.chunker = chunker 
        self._mutual_exclusion_index = self._build_mutual_exclusion_index()

    def _build_mutual_exclusion_index(self) -> Dict[str, Set[Tuple[str, str]]]:
        index = defaultdict(set)
        for a, b in self.mutual_exclusion:
            index[a].add((a, b))
            index[b].add((a, b))
        return index

    def match(
        self,
        text: str,
    ) -> List[RegexMatchResult]:
        results = []
        state = RuleExecutionState(self.rules, self._mutual_exclusion_index)

        for chunk_id, chunk in self.chunker.chunks(text):
            for rule in self.rule_selector.select(chunk_id, state):
                matches = [m for m in rule.match(chunk)]
                
                for match in matches:
                    results.append(RegexMatchResult(rule.id, chunk, match.groupdict()))
                    state.disable_group(rule.id)
                
                if matches and self.stop_on_first_match:
                    break

            if state.is_exhausted():
                break

        return results


class LineChunksRegexMatcher(RegexMatcherTemplate):
    def __init__(self, rules: List[RegexMatcherRule], mutual_exclusion: Set[Tuple[str, str]]) -> None:
        super().__init__(
            rules, 
            mutual_exclusion, 
            stop_on_first_match=True, 
            rule_selector=LineChunksRuleSelector(), 
            chunker=LineChunker()
        )
        

class FullTextRegexMatcher(RegexMatcherTemplate):
    def __init__(self, rules: List[RegexMatcherRule], mutual_exclusion: Set[Tuple[str, str]]) -> None:
        super().__init__(
            rules, 
            mutual_exclusion, 
            stop_on_first_match=False, 
            rule_selector=FullTextRuleSelector(), 
            chunker=FullTextChunker()
        )