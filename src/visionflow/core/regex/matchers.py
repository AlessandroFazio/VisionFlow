# visionflow/regex/matchers.py

from typing import List, Set, Tuple
from visionflow.core.regex.base import RegexMatcherBase, RegexMatchResult, RegexMatcherRule


class RegexMatcherByLine(RegexMatcherBase):
    def __init__(
        self,
        rules: List[RegexMatcherRule],
        global_exclusive: List[Tuple[str, ...]] = [],
        line_exclusive: List[Tuple[str, ...]] = []
    ):
        super().__init__(rules)
        self.global_exclusive = global_exclusive
        self.line_exclusive = line_exclusive

    def match(self, text: str) -> List[RegexMatchResult]:
        results: List[RegexMatchResult] = []
        matched_global: Set[str] = set()

        for lineno, line in enumerate(text.splitlines()):
            matched_line: Set[str] = set()

            for rule in self.rules:
                if rule.target_linenos and lineno not in rule.target_linenos:
                    continue
                if any(rule.id in group and any(r in matched_line for r in group) for group in self.line_exclusive):
                    continue
                if any(rule.id in group and any(r in matched_global for r in group) for group in self.global_exclusive):
                    continue

                for match in rule.match(line):
                    results.append(
                        RegexMatchResult(
                            rule_id=rule.id,
                            text=line,
                            lineno=lineno,
                            matches=match.groupdict()
                        )
                    )
                    matched_line.add(rule.id)
                    matched_global.add(rule.id)

        return results


class RegexMatcherFullText(RegexMatcherBase):
    def __init__(self, rules: List[RegexMatcherRule], mutually_exclusive: List[Tuple[str, ...]] = []):
        super().__init__(rules)
        self.mutually_exclusive = mutually_exclusive

    def match(self, text: str) -> List[RegexMatchResult]:
        results: List[RegexMatchResult] = []
        matched: Set[str] = set()

        for rule in self.rules:
            if rule.id in matched:
                continue
            if any(rule.id in group and any(r in matched for r in group) for group in self.mutually_exclusive):
                continue

            for match in rule.match(text):
                results.append(
                    RegexMatchResult(
                        rule_id=rule.id,
                        text=text,
                        lineno=None,
                        matches=match.groupdict()
                    )
                )
                matched.add(rule.id)

        return results
