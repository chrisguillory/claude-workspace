"""HTTP bridge for CLI client access to BrowserService.

FastAPI app factory that creates endpoints for single tool calls and
batch pipelines. Created inside the MCP server's lifespan with the
shared BrowserService injected via closure (DI pattern from CLAUDE.md).

Dispatch uses BrowserService.tool_registry — no getattr, no MCP protocol
in the HTTP path.

Concurrency: The asyncio.Lock serializes CLI requests against each other.
MCP tools do NOT acquire this lock — FastMCP stdio is inherently sequential,
and Claude Code serializes tool calls (waits for response before sending next).
The lock only guards CLI-vs-CLI interleaving, not MCP-vs-CLI.
"""

from __future__ import annotations

__all__ = [
    'BridgeError',
    'PipelineRequest',
    'PipelineResponse',
    'PipelineStep',
    'StepResult',
    'ToolRequest',
    'ToolResponse',
    'create_bridge_app',
]

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import fastapi
import pydantic
from cc_lib.schemas.base import ClosedModel

from .service import BrowserService

logger = logging.getLogger(__name__)

type ToolStatus = Literal['ok', 'error']
type StepStatus = Literal['ok', 'error', 'skipped']
type PipelineStatus = Literal['completed', 'partial']
type OnErrorPolicy = Literal['stop', 'continue']


# -- Request/Response models --


class BridgeError(ClosedModel):
    """Structured error from bridge dispatch."""

    type: str
    message: str


class ToolRequest(ClosedModel):
    """Single tool invocation request."""

    tool: str
    params: Mapping[str, Any] = {}  # strict_typing_linter.py: loose-typing — tool params vary per tool


class ToolResponse(ClosedModel):
    """Single tool invocation response."""

    status: ToolStatus
    result: Any = None  # strict_typing_linter.py: loose-typing — tool results vary per tool
    elapsed_ms: int = 0
    error: BridgeError | None = None


class PipelineStep(ClosedModel):
    """One step in a pipeline batch."""

    tool: str
    params: Mapping[str, Any] = {}  # strict_typing_linter.py: loose-typing — tool params vary per tool


class PipelineRequest(ClosedModel):
    """Batch pipeline request — ordered sequence of tool calls."""

    steps: Sequence[PipelineStep]
    on_error: OnErrorPolicy = 'stop'


class StepResult(ClosedModel):
    """Result of one pipeline step."""

    step: int
    tool: str
    status: StepStatus
    elapsed_ms: int = 0
    result: Any = None  # strict_typing_linter.py: loose-typing — tool results vary per tool
    error: BridgeError | None = None


class PipelineResponse(ClosedModel):
    """Batch pipeline response with per-step results."""

    status: PipelineStatus
    completed: int
    total: int
    elapsed_ms: int
    results: Sequence[StepResult]


# -- Factory --


def create_bridge_app(service: BrowserService, lock: asyncio.Lock) -> fastapi.FastAPI:
    """Factory: create HTTP bridge with service injected via closure."""
    app = fastapi.FastAPI(title='Selenium Browser Automation HTTP Bridge')

    @app.post('/tool')
    async def tool_endpoint(request: ToolRequest) -> ToolResponse:
        """Execute a single tool call."""
        t0 = time.perf_counter()
        async with lock:
            try:
                result = await BrowserService.tool_registry.dispatch(
                    request.tool,
                    service,
                    **request.params,
                )
                elapsed = int((time.perf_counter() - t0) * 1000)
                return ToolResponse(status='ok', result=result, elapsed_ms=elapsed)
            except (ValueError, pydantic.ValidationError) as e:
                elapsed = int((time.perf_counter() - t0) * 1000)
                return ToolResponse(
                    status='error',
                    elapsed_ms=elapsed,
                    error=BridgeError(type=type(e).__name__, message=str(e)),
                )
            except (
                Exception
            ) as e:  # exception_safety_linter.py: swallowed-exception — HTTP boundary handler returns error response
                elapsed = int((time.perf_counter() - t0) * 1000)
                logger.exception('Tool %s failed', request.tool)
                return ToolResponse(
                    status='error',
                    elapsed_ms=elapsed,
                    error=BridgeError(type=type(e).__name__, message=str(e)),
                )

    @app.post('/pipeline')
    async def pipeline_endpoint(request: PipelineRequest) -> PipelineResponse:
        """Execute an ordered sequence of tool calls."""
        t0 = time.perf_counter()
        results: list[StepResult] = []
        completed = 0

        async with lock:
            for i, step in enumerate(request.steps):
                step_t0 = time.perf_counter()
                try:
                    result = await BrowserService.tool_registry.dispatch(
                        step.tool,
                        service,
                        **step.params,
                    )
                    step_elapsed = int((time.perf_counter() - step_t0) * 1000)
                    results.append(
                        StepResult(
                            step=i,
                            tool=step.tool,
                            status='ok',
                            elapsed_ms=step_elapsed,
                            result=result,
                        )
                    )
                except Exception as e:  # exception_safety_linter.py: swallowed-exception — pipeline step returns error result, continues or stops per on_error
                    step_elapsed = int((time.perf_counter() - step_t0) * 1000)
                    results.append(
                        StepResult(
                            step=i,
                            tool=step.tool,
                            status='error',
                            elapsed_ms=step_elapsed,
                            error=BridgeError(type=type(e).__name__, message=str(e)),
                        )
                    )

                # completed = attempted (not skipped) — both ok and error count
                completed += 1

                if request.on_error == 'stop' and results[-1].status == 'error':
                    results.extend(
                        StepResult(step=j, tool=request.steps[j].tool, status='skipped')
                        for j in range(i + 1, len(request.steps))
                    )
                    break

        total_elapsed = int((time.perf_counter() - t0) * 1000)
        status: PipelineStatus = 'completed' if completed == len(request.steps) else 'partial'

        return PipelineResponse(
            status=status,
            completed=completed,
            total=len(request.steps),
            elapsed_ms=total_elapsed,
            results=results,
        )

    return app
