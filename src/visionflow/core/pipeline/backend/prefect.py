
from prefect import Flow, Task, task
from prefect.tasks import task_input_hash
from prefect.futures import PrefectFuture

from visionflow.core.pipeline.backend.base import S, CompilerBase, DispatcherResultBase, StepDispatcherBase
from visionflow.core.pipeline.base import CompositeStep, StepBase, StepRunContext
from visionflow.core.pipeline.backend.base import BackendBase
from visionflow.core.pipeline.backend.base import PipelineProxyBase, StepProxyBase
from visionflow.core.pipeline.base import StepBase


class PrefectStepProxy(StepProxyBase):
    def __init__(self, wrapped: StepBase, executor: Task) -> None:
        super().__init__(wrapped=wrapped)
        self._executor = executor
    
    def process(self, context: StepRunContext) -> StepRunContext:
        return self._executor(context)
    
    
class PrefectPipelineProxy(PipelineProxyBase):
    def __init__(self, wrapped: StepBase, executor: Flow) -> None:
        super().__init__(wrapped=wrapped)
        self._executor = executor
    
    def process(self, context: StepRunContext) -> StepRunContext:
        return self._executor(context)
    
    
class PrefectDispatcherResult(DispatcherResultBase):
    def __init__(self, future: PrefectFuture) -> None:
        self._future = future
        
    def get(self) -> StepRunContext:
        return self._future.result()
 

class PrefectDispatcher(StepDispatcherBase):
    def submit(self, step: StepBase, context: StepRunContext) -> DispatcherResultBase:
        if isinstance(step, (PrefectStepProxy, PrefectPipelineProxy)):
            future = step._executor.submit(context)
            return PrefectDispatcherResult(future)
        raise TypeError("")


class PrefectCompiler(CompilerBase):
    def compile(self, step: S) -> S:
        if isinstance(step, (PrefectStepProxy, PrefectPipelineProxy)):
            return step
        
        if isinstance(step, CompositeStep):
            compiled_steps = [self.compile(s) for s in step.steps()]
            rebuilt = step.rebuild_with_steps(compiled_steps)
            task = self._compile_task(rebuilt)
            return PrefectStepProxy(rebuilt, task)
        
        else:
            task = self._compile_task(step)
            return PrefectStepProxy(step, task)
            
    def _compile_task(self, step: StepBase) -> Task:
        opts = step.runtime_options
        kwargs = {
            "name": step.name
        }

        if opts.cache_expiration:
            kwargs.update({
                "cache_key_fn": task_input_hash,
                "cache_expiration": opts.cache_expiration
            })

        if opts.retries:
            kwargs.update({
                "retries": opts.retries,
                "retry_delay_seconds": opts.retry_delay_seconds or 0,
            })

        if opts.concurrency_limit:
            kwargs["task_run_concurrency_limit"] = opts.concurrency_limit

        @task(**kwargs)
        def compiled_task(context: StepRunContext) -> StepRunContext:
            return step.process(context)

        return compiled_task


class PrefectBackend(BackendBase):
    def __init__(self) -> None:
        super().__init__(
            compiler=PrefectCompiler(),
            dispatcher=PrefectDispatcher()
        )