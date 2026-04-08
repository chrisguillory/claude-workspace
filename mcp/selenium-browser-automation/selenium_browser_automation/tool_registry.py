from __future__ import annotations

__all__ = [
    'ToolRegistry',
]

from collections.abc import Callable, Iterator
from typing import Any


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

    def register_tool(
        self, fn: Callable[..., Any]
    ) -> Callable[
        ..., Any
    ]:  # strict_typing_linter.py: loose-typing — generic decorator factory, accepts arbitrary callables
        """Decorator: register a method as a dispatchable tool."""
        self._tools[fn.__name__] = fn
        return fn

    async def dispatch(
        self, name: str, service: Any, **params: Any
    ) -> Any:  # strict_typing_linter.py: loose-typing — generic dispatch, types vary per tool
        """Call a registered tool by name on the given service instance."""
        fn = self._tools.get(name)
        if fn is None:
            raise ValueError(f'Unknown tool: {name}')
        return await fn(service, **params)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)
