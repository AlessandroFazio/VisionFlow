from visionflow.core.pipeline.backend.base import S, BackendBase, CompilerBase, DispatcherResultBase, StepDispatcherBase
from visionflow.core.pipeline.base import StepBase, StepRunContext


class PythonCompiler(CompilerBase):
    def compile(self, step: S) -> S:
        return step


class PythonDispatcherResult(DispatcherResultBase):
    def __init__(self, context: StepRunContext) -> None:
        self._context = context

    def get(self) -> StepRunContext:
        return self._context 


class PythonDispatcher(StepDispatcherBase):
    def submit(self, step: StepBase, context: StepRunContext) -> DispatcherResultBase:
        context = step.process(context)
        return PythonDispatcherResult(context)


class PythonBackend(BackendBase):
    def __init__(self) -> None:
        super().__init__(
            compiler=PythonCompiler(),
            dispatcher=PythonDispatcher()
        )

    