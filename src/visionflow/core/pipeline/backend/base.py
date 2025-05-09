from abc import ABC, abstractmethod
from typing import Iterable, Iterator, TypeVar

from visionflow.core.pipeline.base import StepBase, StepRunContext
from visionflow.core.pipeline.pipeline import Pipeline
from visionflow.typing.proxy import ProxyMixin

S = TypeVar('S', bound=StepBase)


class StepProxyBase(ProxyMixin[StepBase], StepBase):
    def __init__(self, wrapped: StepBase) -> None:
        StepBase.__init__(self, name=f"{self.__class__.__name__}[{wrapped.name}]")
        ProxyMixin.__init__(self, wrapped)
        

class PipelineProxyBase(ProxyMixin[Pipeline], Pipeline):
    def __init__(self, wrapped: Pipeline) -> None:
        Pipeline.__init__(self, name=f"{self.__class__.__name__}[{wrapped.name}]")
        ProxyMixin.__init__(self, wrapped)


class CompilerBase(ABC):
    """Compiles a Pipeline or Step into a backend-specific executable form."""
    @abstractmethod
    def compile(self, step: S) -> S:
        pass
    

class DispatcherResultBase:
    @abstractmethod
    def get(self) -> StepRunContext:
        pass


class StepDispatcherBase(ABC):
    @abstractmethod
    def submit(self, step: StepBase, context: StepRunContext) -> DispatcherResultBase:
        pass


class BackendBase(ABC):
    def __init__(self, compiler: CompilerBase, dispatcher: StepDispatcherBase) -> None:
        self._compiler = compiler
        self._dispatcher = dispatcher

    def compile(self, pipeline: Pipeline) -> Pipeline:
        return self._compiler.compile(pipeline)
    
    def run(self, pipeline: Pipeline, context: StepRunContext) -> StepRunContext:
        compiled = self.compile(pipeline)
        return compiled.run(context)
    
    def run_batch(self, pipeline: Pipeline, contexts: Iterable[StepRunContext]) -> Iterator[StepRunContext]:
        compiled = self.compile(pipeline)
        results = [self._dispatcher.submit(compiled, ctx) for ctx in contexts]
        for r in results:
            yield r.get()
    
    def dispatcher(self) -> StepDispatcherBase:
        return self._dispatcher