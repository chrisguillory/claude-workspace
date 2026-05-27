from __future__ import annotations

__all__ = [
    'ToolRegistry',
]

import typing
from collections.abc import Callable, Iterator
from typing import Any

import pydantic


class ToolRegistry:
    """Typed dispatch registry for service methods.

    Decorates async methods on a service class, then dispatches string tool names
    to those methods via the bridge HTTP layer. Same pattern as Flask's
    app.view_functions, FastAPI's app.routes, and Click's group.commands.

    Usage::

        class MyService:
            tool_registry: ClassVar[ToolRegistry] = ToolRegistry()

            @tool_registry.register_tool
            async def click(self, css_selector: str) -> None: ...


        # Bridge dispatch:
        await MyService.tool_registry.dispatch('click', service_instance, css_selector='#btn')
    """

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}

    def register_tool[F: Callable[..., Any]](self, fn: F) -> F:
        """Decorator: register a method as a dispatchable tool.

        Generic ``F`` preserves the decorated function's full signature so
        mypy sees e.g. ``async def navigate(...) -> NavigationResult``
        after decoration, not ``Callable[..., Any]``.
        """
        self._tools[fn.__name__] = fn
        return fn

    async def dispatch(
        self, name: str, service: Any, **params: Any
    ) -> Any:  # strict_typing_linter.py: loose-typing — generic dispatch, types vary per tool
        """Call a registered tool by name on the given service instance.

        Validates each kwarg against the target function's declared type annotation
        via ``pydantic.TypeAdapter``. The bridge receives JSON which lands as
        Python primitives/dicts; this step coerces them into the typed objects the
        service expects (e.g. ``WindowSize`` from a ``{"width": ..., "height": ...}``
        dict). Mirrors what FastMCP does on the MCP side, so the CLI/bridge path
        and the MCP path arrive at the service with the same typed inputs.

        ``pydantic.ValidationError`` raised here is caught and returned as a
        structured ``BridgeError`` by ``bridge.tool_endpoint``.
        """
        fn = self._tools.get(name)
        if fn is None:
            raise ValueError(f'Unknown tool: {name}')
        hints = typing.get_type_hints(fn)
        coerced = {
            k: (pydantic.TypeAdapter(hints[k]).validate_python(v) if k in hints else v) for k, v in params.items()
        }
        return await fn(service, **coerced)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
