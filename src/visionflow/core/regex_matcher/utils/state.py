from typing import Dict, List, Set, Tuple

from visionflow.core.regex_matcher.base import RegexMatcherRule


class RuleExecutionState:
    def __init__(
        self, 
        rules: List[RegexMatcherRule], 
        mutual_exclusion_index: Dict[str, Set[Tuple[str, str]]]
    ) -> None:
        self.enabled_rules: Set[str] = {r.id for r in rules}
        self.mutual_exclusion_index = mutual_exclusion_index

    def is_enabled(self, rule_id: str) -> bool:
        return rule_id in self.enabled_rules

    def disable(self, rule_id: str):
        self.enabled_rules.discard(rule_id)

    def disable_group(self, rule_id: str) -> None:
        for a, b in self.mutual_exclusion_index.get(rule_id, []):
            self.disable(a if b == rule_id else b)

    def is_exhausted(self) -> bool:
        return not self.enabled_rules