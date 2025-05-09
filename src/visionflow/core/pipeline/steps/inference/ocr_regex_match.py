import dataclasses
from typing import Any, Dict, Optional
from visionflow.core.pipeline.base import RuntimeOptions, StepBase, StepRunContext
from visionflow.core.pipeline.utils.providers import TextProviderBase
from visionflow.core.regex_matcher.base import RegexMatcherBase


class OcrRegexMatchStep(StepBase):
    def __init__(
        self, 
        matcher: RegexMatcherBase, 
        provider: TextProviderBase,
        runtime_options: Optional[RuntimeOptions]=None, 
        tags: Optional[Dict[str, Any]]=None
    ) -> None:
        super().__init__(runtime_options=runtime_options, tags=tags)
        self.matcher = matcher
        self.provider = provider
        
    def process(self, context: StepRunContext) -> StepRunContext:
        text = self.provider(context.exchange)
        results = self.matcher.match(text)
        exchange = dataclasses.replace(context.exchange, ocr_regex_matches=results)
        return dataclasses.replace(context, exchange=exchange)