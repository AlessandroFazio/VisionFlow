import dataclasses
from visionflow.core.pipeline.base import Exchange, PipelineContext, StepBase
from visionflow.core.pipeline.utils.providers import TextProviderBase
from visionflow.core.regex_matcher.base import RegexMatcherBase


class OcrRegexMatchStep(StepBase):
    def __init__(self, matcher: RegexMatcherBase, provider: TextProviderBase) -> None:
        super().__init__()
        self.matcher = matcher
        self.provider = provider
        
    def process(self, context: PipelineContext, exchange: Exchange) -> Exchange:
        text = self.provider(exchange)
        results = self.matcher.match(text)
        return dataclasses.replace(exchange, ocr_regex_matches=results)