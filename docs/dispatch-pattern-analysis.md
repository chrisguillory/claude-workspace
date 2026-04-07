---
title: "Dispatch Pattern Analysis: String Tool Name to Service Method"
date: 2026-04-05
---

# Dispatch Pattern Analysis: String Tool Name to Service Method

## Problem Statement

A `BrowserService` class exposes 31 public async methods (`navigate`, `click`, `screenshot`, etc.). An HTTP bridge receives JSON like `{"tool": "click", "params": {"css_selector": "#btn"}}` and must dispatch the string tool name to the correct method. The naive approach -- `getattr(service, tool_name)` -- is stringly-typed with no static analysis support.

**Requirements:**

1. Map string tool names to service methods
2. Type-safe (mypy-friendly)
3. Auto-update when methods are added or removed
4. Work with async methods
5. Pythonic and clean

---

## Pattern 1: Auto-Built Dispatch Dict at Init

Build the dispatch table once at construction time by reflecting over public methods.

```python
from collections.abc import Callable, Coroutine
from typing import Any

class BrowserService:
    async def click(self, css_selector: str) -> dict[str, Any]: ...
    async def navigate(self, url: str) -> dict[str, Any]: ...
    async def screenshot(self, filename: str) -> dict[str, Any]: ...
    # ... 28 more methods

type AsyncMethod = Callable[..., Coroutine[Any, Any, Any]]

class Dispatcher:
    def __init__(self, service: BrowserService) -> None:
        self._dispatch: dict[str, AsyncMethod] = {
            name: getattr(service, name)
            for name in dir(service)
            if not name.startswith("_") and callable(getattr(service, name))
        }

    async def handle(self, tool: str, params: dict[str, Any]) -> Any:
        method = self._dispatch.get(tool)
        if method is None:
            raise ValueError(f"Unknown tool: {tool}")
        return await method(**params)
```

**Call site:**

```python
dispatcher = Dispatcher(service)
result = await dispatcher.handle("click", {"css_selector": "#btn"})
```

| Criterion | Assessment |
|-----------|-----------|
| mypy understands types? | Partially. The dict values are typed as `Callable[..., Coroutine]`, so mypy knows the return is awaitable, but individual method signatures are erased. The `getattr` call in `__init__` is typed `Any`. |
| Auto-updates on new methods? | Yes. Any new public method on `BrowserService` appears automatically. |
| Call site clarity? | Clean. Single `handle()` entry point. |
| Runtime overhead? | One-time `dir()` + `getattr` scan at init. Negligible. Dict lookup at O(1) per call. |
| Risks? | Exposes ALL public methods, including ones not intended as tools (e.g., utility methods, properties). No allowlist means the blast radius is the entire public surface. |

**Verdict:** This is `getattr` with extra steps. The reflection still happens, just at init instead of per-call. It gains nothing for type safety -- mypy sees `dict[str, Callable[..., Coroutine]]`, losing all per-method parameter types. The "auto-update" is a double-edged sword: any public method, including helpers not meant as tools, gets exposed to the HTTP bridge.

---

## Pattern 2: Decorator-Based Registration

Methods opt in to dispatch via a decorator that populates a class-level registry.

```python
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

# Module-level registry populated at class definition time
_tool_registry: dict[str, str] = {}  # tool_name -> method_name

def tool(name: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Register a method as a dispatchable tool."""
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        _tool_registry[name] = fn.__name__
        return fn  # Return unmodified -- no wrapper overhead
    return decorator


class BrowserService:
    @tool("click")
    async def click(self, css_selector: str) -> dict[str, Any]:
        ...

    @tool("navigate")
    async def navigate(self, url: str, fresh_browser: bool = False) -> dict[str, Any]:
        ...

    @tool("screenshot")
    async def screenshot(self, filename: str, full_page: bool = False) -> dict[str, Any]:
        ...

    # Helper method -- NOT decorated, NOT dispatchable
    def _build_chrome_options(self) -> Any:
        ...


class Dispatcher:
    def __init__(self, service: BrowserService) -> None:
        self._dispatch: dict[str, Callable[..., Any]] = {
            tool_name: getattr(service, method_name)
            for tool_name, method_name in _tool_registry.items()
        }

    async def handle(self, tool: str, params: dict[str, Any]) -> Any:
        method = self._dispatch.get(tool)
        if method is None:
            raise ValueError(f"Unknown tool: {tool}")
        return await method(**params)
```

**Call site:**

```python
dispatcher = Dispatcher(service)
result = await dispatcher.handle("click", {"css_selector": "#btn"})
```

| Criterion | Assessment |
|-----------|-----------|
| mypy understands types? | Partially. The decorator preserves the method's signature (it returns the function unchanged, typed with `ParamSpec`). But the dispatch dict erases signatures to `Callable[..., Any]`. Individual decorated methods remain fully typed when called directly. |
| Auto-updates on new methods? | No -- you must add `@tool("name")` to each new method. Forgetting the decorator means the method is silently unreachable via dispatch. |
| Call site clarity? | Clean. The decorator clearly marks which methods are tools and what string maps to them. Reading the class tells you the dispatch table. |
| Runtime overhead? | Zero per-call. Decorator runs once at class definition. Dict lookup at dispatch. |
| Risks? | Tool name can diverge from method name (`@tool("nav")` on `navigate()`). Duplication between decorator arg and method name. |

**Variant -- derive name from the method automatically:**

```python
def tool(fn: Callable[P, R]) -> Callable[P, R]:
    """Register using the method's own name. No string argument needed."""
    _tool_registry[fn.__name__] = fn.__name__
    return fn

class BrowserService:
    @tool
    async def click(self, css_selector: str) -> dict[str, Any]: ...
```

This eliminates name duplication but loses the ability to remap names.

**Verdict:** This is the FastAPI/Flask model. It is explicit, Pythonic, and widely understood. The cost is that every new tool method needs a decorator line. But that is also the benefit: dispatch membership is an intentional, visible decision at the point of definition. The decorator is also a natural place to attach metadata (descriptions, parameter schemas) if needed later.

---

## Pattern 3: `__init_subclass__` / Metaclass Registry

Use Python's class machinery to auto-register methods marked with a sentinel attribute.

```python
from collections.abc import Callable
from typing import Any

class ToolService:
    """Base class that auto-discovers methods marked as tools."""
    _tool_registry: dict[str, str] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, method in vars(cls).items():
            if getattr(method, "_is_tool", False):
                cls._tool_registry[name] = name

    def get_dispatch_table(self) -> dict[str, Callable[..., Any]]:
        return {
            name: getattr(self, name)
            for name in self._tool_registry
        }


def tool(fn: Callable[..., Any]) -> Callable[..., Any]:
    fn._is_tool = True  # type: ignore[attr-defined]
    return fn


class BrowserService(ToolService):
    @tool
    async def click(self, css_selector: str) -> dict[str, Any]: ...

    @tool
    async def navigate(self, url: str) -> dict[str, Any]: ...

    # Not a tool -- no decorator
    def _build_chrome_options(self) -> Any: ...


class Dispatcher:
    def __init__(self, service: BrowserService) -> None:
        self._dispatch = service.get_dispatch_table()

    async def handle(self, tool_name: str, params: dict[str, Any]) -> Any:
        method = self._dispatch.get(tool_name)
        if method is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await method(**params)
```

| Criterion | Assessment |
|-----------|-----------|
| mypy understands types? | No. `fn._is_tool = True` requires `# type: ignore`. The dispatch table is `dict[str, Callable[..., Any]]`. mypy cannot reason about the sentinel attribute. |
| Auto-updates on new methods? | Requires `@tool` decorator. Same opt-in as Pattern 2 but with more machinery. |
| Call site clarity? | Identical to Pattern 2 at the call site. But the registration mechanism is hidden in `__init_subclass__`, which most developers find surprising. |
| Runtime overhead? | Runs once at class definition (not even at instantiation). Negligible. |
| Risks? | Inheritance creates shared mutable state. If `BrowserServiceV2(BrowserService)` adds tools, the parent's `_tool_registry` is mutated. Must copy-on-write per subclass. The `# type: ignore` is a code smell. |

**Verdict:** Over-engineered for this use case. `__init_subclass__` shines when you have a plugin system with many subclasses registering themselves (like Django admin, pytest fixtures). Here we have one service class. Pattern 2's simple decorator achieves the same result with less indirection and better mypy support.

---

## Pattern 4: `functools.singledispatch` or Custom Type-Based Dispatch

`singledispatch` dispatches on the *type* of the first argument, not on a *string value*. It does not apply directly to our problem (we dispatch on a string tool name, not on argument types). But we can build an analogous pattern.

```python
import functools
from typing import Any

# singledispatch dispatches on TYPE, not VALUE -- won't work directly
@functools.singledispatch
async def dispatch(tool: object, params: dict[str, Any]) -> Any:
    raise ValueError(f"Unknown tool type: {type(tool)}")

# You'd need a different type per tool -- absurd
class ClickTool: pass
class NavigateTool: pass

@dispatch.register(ClickTool)
async def _(tool: ClickTool, params: dict[str, Any]) -> Any:
    return await service.click(**params)
```

This is clearly wrong for our use case. `singledispatch` solves a different problem (polymorphic dispatch on argument type). Including it here only to demonstrate why it does not fit.

**Could we build a custom value-based dispatcher?**

```python
from collections.abc import Callable, Coroutine
from typing import Any

type AsyncHandler = Callable[..., Coroutine[Any, Any, Any]]

class ValueDispatcher:
    def __init__(self) -> None:
        self._registry: dict[str, AsyncHandler] = {}

    def register(self, name: str) -> Callable[[AsyncHandler], AsyncHandler]:
        def decorator(fn: AsyncHandler) -> AsyncHandler:
            self._registry[name] = fn
            return fn
        return decorator

    async def dispatch(self, name: str, **kwargs: Any) -> Any:
        handler = self._registry.get(name)
        if handler is None:
            raise ValueError(f"Unknown: {name}")
        return await handler(**kwargs)

tool_dispatch = ValueDispatcher()

@tool_dispatch.register("click")
async def click(css_selector: str) -> dict[str, Any]: ...

@tool_dispatch.register("navigate")
async def navigate(url: str) -> dict[str, Any]: ...
```

This works but forces tools to be module-level functions, not methods on a service class. Combining it with a service class reintroduces the same binding problems solved by Pattern 2.

| Criterion | Assessment |
|-----------|-----------|
| mypy understands types? | `singledispatch` has good mypy support for type-based dispatch, but it is the wrong tool here. Custom `ValueDispatcher` loses method signatures just like other dict approaches. |
| Auto-updates on new methods? | Requires explicit registration via decorator. |
| Call site clarity? | Awkward. Separates tool functions from the service class they logically belong to. |
| Runtime overhead? | Dict lookup. Same as everything else. |

**Verdict:** `singledispatch` is the wrong tool. A custom `ValueDispatcher` is just Pattern 2 reinvented as a standalone object. No advantage.

---

## Pattern 5: Match Statement (Python 3.10+)

Explicit structural pattern matching. Every tool gets a `case` branch.

```python
from typing import Any

class Dispatcher:
    def __init__(self, service: BrowserService) -> None:
        self._service = service

    async def handle(self, tool: str, params: dict[str, Any]) -> Any:
        match tool:
            case "click":
                return await self._service.click(**params)
            case "navigate":
                return await self._service.navigate(**params)
            case "screenshot":
                return await self._service.screenshot(**params)
            case "get_aria_snapshot":
                return await self._service.get_aria_snapshot(**params)
            case "get_page_text":
                return await self._service.get_page_text(**params)
            case "get_interactive_elements":
                return await self._service.get_interactive_elements(**params)
            case "scroll":
                return await self._service.scroll(**params)
            case "type_text":
                return await self._service.type_text(**params)
            case "press_key":
                return await self._service.press_key(**params)
            case "hover":
                return await self._service.hover(**params)
            case "wait_for_selector":
                return await self._service.wait_for_selector(**params)
            case "wait_for_network_idle":
                return await self._service.wait_for_network_idle(**params)
            case "execute_javascript":
                return await self._service.execute_javascript(**params)
            case "get_page_html":
                return await self._service.get_page_html(**params)
            case "get_console_logs":
                return await self._service.get_console_logs(**params)
            case "save_profile_state":
                return await self._service.save_profile_state(**params)
            case "navigate_with_profile_state":
                return await self._service.navigate_with_profile_state(**params)
            case "export_chrome_profile_state":
                return await self._service.export_chrome_profile_state(**params)
            case "export_har":
                return await self._service.export_har(**params)
            case "get_resource_timings":
                return await self._service.get_resource_timings(**params)
            case "capture_web_vitals":
                return await self._service.capture_web_vitals(**params)
            case "resize_window":
                return await self._service.resize_window(**params)
            case "set_blocked_urls":
                return await self._service.set_blocked_urls(**params)
            case "sleep":
                return await self._service.sleep(**params)
            case "download_resource":
                return await self._service.download_resource(**params)
            case "configure_proxy":
                return await self._service.configure_proxy(**params)
            case "clear_proxy":
                return await self._service.clear_proxy(**params)
            case "list_chrome_profiles":
                return await self._service.list_chrome_profiles(**params)
            case "get_focusable_elements":
                return await self._service.get_focusable_elements(**params)
            case "get_visual_tree":
                return await self._service.get_visual_tree(**params)
            case "get_file_info":
                return await self._service.get_file_info(**params)
            case _:
                raise ValueError(f"Unknown tool: {tool}")
```

| Criterion | Assessment |
|-----------|-----------|
| mypy understands types? | **Best of all patterns.** Each branch is a direct method call. mypy validates every `**params` spread against the actual method signature -- if `click` requires `css_selector: str`, mypy knows. However, since `params` is `dict[str, Any]`, this advantage is theoretical unless you also validate params per-branch. |
| Auto-updates on new methods? | **No.** You must manually add a `case` for every new method. Forgetting a case means silent 404 at runtime. This is the worst pattern for the "31 methods" scale. |
| Call site clarity? | Extremely explicit. A reader sees every mapping at a glance. |
| Runtime overhead? | CPython 3.10+ optimizes match statements, but 31 cases is still a linear scan in the worst case. The real cost is not CPU but developer time maintaining the block. |
| Risks? | 31 nearly-identical lines of boilerplate. Adding method 32 means editing the dispatcher. Violates DRY. Prone to copy-paste errors. |

**Verdict:** Maximum type safety and readability per-branch, but catastrophic maintenance burden at 31 methods. This pattern works for 3-5 cases. At 31, it is a liability. Every new tool requires a two-place edit (service class + match block), and there is no mechanism to detect a missing case.

---

## Pattern 6: Enum-Based Dispatch

Define tool names as an Enum, map enum members to methods.

```python
from enum import Enum
from collections.abc import Callable, Coroutine
from typing import Any

type AsyncMethod = Callable[..., Coroutine[Any, Any, Any]]


class Tool(str, Enum):
    CLICK = "click"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    GET_ARIA_SNAPSHOT = "get_aria_snapshot"
    GET_PAGE_TEXT = "get_page_text"
    SCROLL = "scroll"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    HOVER = "hover"
    # ... 22 more members


class Dispatcher:
    def __init__(self, service: BrowserService) -> None:
        self._dispatch: dict[Tool, AsyncMethod] = {
            Tool.CLICK: service.click,
            Tool.NAVIGATE: service.navigate,
            Tool.SCREENSHOT: service.screenshot,
            Tool.GET_ARIA_SNAPSHOT: service.get_aria_snapshot,
            Tool.GET_PAGE_TEXT: service.get_page_text,
            Tool.SCROLL: service.scroll,
            Tool.TYPE_TEXT: service.type_text,
            Tool.PRESS_KEY: service.press_key,
            Tool.HOVER: service.hover,
            # ... 22 more entries
        }

    async def handle(self, tool_name: str, params: dict[str, Any]) -> Any:
        try:
            tool = Tool(tool_name)  # Validates string against known values
        except ValueError:
            raise ValueError(f"Unknown tool: {tool_name}")
        return await self._dispatch[tool](**params)
```

| Criterion | Assessment |
|-----------|-----------|
| mypy understands types? | The Enum itself is fully typed. `Tool("click")` is type-safe. But the dispatch dict still erases method signatures to `Callable[..., Coroutine]`. You could detect a missing mapping at runtime via `assert set(Tool) == set(dispatch.keys())`. |
| Auto-updates on new methods? | **Three-place edit:** add enum member, add service method, add dict entry. Worst maintenance burden of all patterns. |
| Call site clarity? | Clean at the call site. The Enum provides validation and IDE autocomplete for internal callers using `Tool.CLICK`. |
| Runtime overhead? | Enum construction from string, then dict lookup. Trivially fast. |
| Risks? | Maximum duplication. Enum member, dict entry, and method must all agree. Adding method 32 is a three-file edit. |

**Variant -- Enum with method references as values:**

```python
class Tool(str, Enum):
    CLICK = "click"
    NAVIGATE = "navigate"

    def bind(self, service: BrowserService) -> AsyncMethod:
        """Bind this enum member to the corresponding service method."""
        return getattr(service, self.value)  # Still uses getattr!

# Usage
tool = Tool("click")
result = await tool.bind(service)(**params)
```

This is elegant but secretly `getattr` again.

**Verdict:** Enums add type safety for the tool *name* (validating it against a fixed set) but do not help with the dispatch *mapping*. The maintenance burden of keeping three artifacts in sync (enum, dict, method) is the highest of any pattern. Good for small fixed sets. Poor for 31 evolving methods.

---

## Pattern 7: Web Framework Patterns

### 7a. FastAPI / Starlette Decorator Registry

FastAPI uses `@app.get("/path")` decorators that call `add_api_route()` to append routes to a list. At request time, Starlette linearly scans routes, calling each route's `matches()` method. This is Pattern 2 (decorator registration) with a list instead of a dict and path-matching instead of string equality.

Key insight from the [Starlette source](https://github.com/encode/starlette/blob/master/starlette/routing.py): routes are stored as `self.routes: list[BaseRoute]` and dispatched via linear scan. For string-equality dispatch (our case), a dict is strictly better.

### 7b. Django URL Routing

Django defines URL patterns in a `urlpatterns` list:

```python
urlpatterns = [
    path("click/", views.click),
    path("navigate/", views.navigate),
]
```

This is a manual dict/list -- Pattern 5 (match) in declarative form. Same maintenance burden: adding a view requires editing both the view module and the URL config.

### 7c. Click / Typer CLI Dispatch

Click uses `@group.command()` decorators -- identical to Pattern 2:

```python
@cli.command()
def click(css_selector: str) -> None: ...

@cli.command()
def navigate(url: str) -> None: ...
```

The group maintains an internal dict of `{name: Command}`. This is the exact same decorator-registry pattern.

**Verdict:** Every major Python framework uses Pattern 2 (decorator-based registration). FastAPI, Flask, Click, Typer, Celery (`@app.task`), pytest (`@pytest.fixture`), Django REST Framework (`@action`). The pattern is battle-tested, universally understood, and Pythonic.

---

## Pattern 8: Hybrid -- Decorator with Exhaustiveness Check

This combines Pattern 2's decorator with a runtime exhaustiveness check, addressing the "forgot to decorate a new method" failure mode.

```python
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_tool_registry: dict[str, str] = {}


def tool(fn: Callable[P, R]) -> Callable[P, R]:
    """Register a method as a dispatchable tool. Name derived from method name."""
    _tool_registry[fn.__name__] = fn.__name__
    return fn


class BrowserService:
    @tool
    async def click(self, css_selector: str) -> dict[str, Any]: ...

    @tool
    async def navigate(self, url: str, fresh_browser: bool = False) -> dict[str, Any]: ...

    @tool
    async def screenshot(self, filename: str, full_page: bool = False) -> dict[str, Any]: ...

    # Internal helper -- intentionally NOT a tool
    def _build_chrome_options(self) -> Any: ...


class Dispatcher:
    def __init__(self, service: BrowserService) -> None:
        self._dispatch: dict[str, Callable[..., Any]] = {
            name: getattr(service, name)
            for name in _tool_registry
        }

    async def handle(self, tool: str, params: dict[str, Any]) -> Any:
        method = self._dispatch.get(tool)
        if method is None:
            raise ValueError(f"Unknown tool: {tool}")
        return await method(**params)

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._dispatch)
```

**The exhaustiveness check -- run at startup or in tests:**

```python
# In test_dispatcher.py
def test_all_public_async_methods_are_registered():
    """Catch methods that forgot @tool."""
    import inspect

    service_methods = {
        name
        for name, method in inspect.getmembers(BrowserService, inspect.isfunction)
        if not name.startswith("_")
        and inspect.iscoroutinefunction(method)
    }
    registered = set(_tool_registry.keys())

    missing = service_methods - registered
    assert not missing, (
        f"Public async methods not registered as tools: {missing}. "
        f"Add @tool decorator or prefix with _ if internal."
    )
```

| Criterion | Assessment |
|-----------|-----------|
| mypy understands types? | Same as Pattern 2. Decorated methods retain signatures for direct calls. Dispatch dict erases to `Callable[..., Any]`. |
| Auto-updates on new methods? | **Effectively yes** -- new methods without `@tool` trigger a test failure. The test acts as a compile-time-equivalent guard. |
| Call site clarity? | Clean. Decorator at definition, simple `handle()` at call site. |
| Runtime overhead? | Dict lookup. Identical to all dict-based approaches. |
| Risks? | The exhaustiveness test catches the common failure mode. Decorator is zero-overhead (no wrapper). |

**Verdict:** This is the recommended approach for your case. It combines the strengths of Pattern 2 (explicit opt-in, zero-overhead decorator, widely understood) with a safety net that catches the one failure mode decorators introduce (forgetting to decorate a new method).

---

## Comparison Matrix

| Pattern | Type Safe | Auto-Update | Maintenance | Async | Pythonic |
|---------|-----------|-------------|-------------|-------|---------|
| 1. Auto-built dict | Poor | Yes (too much) | Low | Yes | Moderate |
| 2. Decorator registry | Good | Opt-in | Low | Yes | Excellent |
| 3. `__init_subclass__` | Poor | Opt-in | Medium | Yes | Over-engineered |
| 4. `singledispatch` | N/A | N/A | N/A | N/A | Wrong tool |
| 5. Match statement | Best | Manual | **Very High** | Yes | Good at small scale |
| 6. Enum dispatch | Moderate | Manual (3 places) | **Very High** | Yes | Moderate |
| 7. Framework patterns | (Same as Pattern 2) | | | | |
| **8. Decorator + test** | **Good** | **Opt-in + safety net** | **Low** | **Yes** | **Excellent** |

---

## Recommendation: Pattern 8 (Decorator + Exhaustiveness Test)

For 31 async methods on a single service class, the clear winner is **Pattern 2 with an exhaustiveness test** (Pattern 8).

**Why:**

1. **Every major Python framework uses this pattern.** FastAPI, Flask, Click, Typer, Celery, pytest. Developers recognize it immediately.

2. **Explicit opt-in is a feature, not a bug.** Not every public method should be exposed to an HTTP bridge. The decorator marks the intentional boundary between "internal method" and "externally dispatchable tool."

3. **The exhaustiveness test closes the gap.** The one weakness of decorator registration -- forgetting to decorate a new method -- is caught by a simple test that compares public async methods against the registry. This runs in CI and fails loudly.

4. **Zero runtime overhead.** The decorator returns the original function unmodified. No wrapper, no `functools.wraps`, no stack frame added per call. Registration happens once at class definition time (import time), not at instantiation.

5. **mypy-friendly where it matters.** Direct calls to `service.click(css_selector="#btn")` are fully type-checked. The dispatch path through `handle()` uses `Callable[..., Any]` -- this is inherent to *any* string-based dispatch and cannot be avoided without code generation.

6. **Clean separation of concerns.** The service class owns business logic. The decorator opts methods into dispatch. The dispatcher owns routing. The test owns completeness verification.

**What the recommendation looks like in practice:**

```python
# browser_service.py
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

_tool_registry: dict[str, str] = {}

def tool(fn: Callable[P, R]) -> Callable[P, R]:
    _tool_registry[fn.__name__] = fn.__name__
    return fn

class BrowserService:
    @tool
    async def click(self, css_selector: str) -> dict[str, Any]: ...

    @tool
    async def navigate(self, url: str, fresh_browser: bool = False) -> dict[str, Any]: ...

    @tool
    async def screenshot(self, filename: str, full_page: bool = False) -> dict[str, Any]: ...

    # ... 28 more @tool methods ...

    # Internal helpers: no decorator, prefixed with _
    def _build_chrome_options(self) -> Any: ...
```

```python
# dispatcher.py
from collections.abc import Callable
from typing import Any

from .browser_service import BrowserService, _tool_registry

class Dispatcher:
    def __init__(self, service: BrowserService) -> None:
        self._dispatch: dict[str, Callable[..., Any]] = {
            name: getattr(service, name) for name in _tool_registry
        }

    async def handle(self, tool: str, params: dict[str, Any]) -> Any:
        method = self._dispatch.get(tool)
        if method is None:
            raise ValueError(f"Unknown tool: {tool}")
        return await method(**params)
```

```python
# test_dispatcher.py
import inspect
from browser_service import BrowserService, _tool_registry

def test_all_public_async_methods_are_registered():
    service_methods = {
        name
        for name, method in inspect.getmembers(BrowserService, inspect.isfunction)
        if not name.startswith("_") and inspect.iscoroutinefunction(method)
    }
    registered = set(_tool_registry.keys())
    missing = service_methods - registered
    assert not missing, f"Unregistered tools: {missing}. Add @tool or prefix with _."
```

---

## Why Not the Others?

| Pattern | Rejection Reason |
|---------|-----------------|
| Auto-built dict (1) | Exposes everything, including non-tool methods. `getattr` in disguise. |
| `__init_subclass__` (3) | Over-engineered for one class. Shared mutable state across subclasses. `# type: ignore` required. |
| `singledispatch` (4) | Dispatches on type, not string value. Wrong abstraction. |
| Match statement (5) | 31 identical `case` branches. Every new method = edit dispatcher. Violates DRY. |
| Enum dispatch (6) | Three-place edit for every new method (enum + dict + method). Maximum duplication. |

---

## A Note on Type Safety at the Dispatch Boundary

No pattern can give mypy full parameter-level type checking at the `handle("click", {"css_selector": "#btn"})` call site. The string `"click"` and the `dict` of params are both dynamic. This is inherent to the problem: an HTTP bridge receives untyped JSON.

The type safety boundary is:

```
HTTP JSON (untyped) --> handle() --> service.click(css_selector="#btn") (fully typed)
                        ^                    ^
                        dispatch boundary    mypy checks this if called directly
```

If you need mypy to validate params per-tool, you would need code generation (generating a typed `handle` overload per tool from the registry) or runtime validation (Pydantic models per tool). That is a separate concern from dispatch itself.

---

## Sources

- [Dictionary Dispatch Pattern in Python](https://martinheinz.dev/blog/90) -- Martin Heinz
- [The Python dictionary dispatch pattern](https://jamesg.blog/2023/08/26/python-dictionary-dispatch) -- James' Coffee Blog
- [How FastAPI path operations work](https://vickiboykis.com/2025/01/14/how-fastapi-path-operations-work/) -- Vicki Boykis
- [Python Registry Pattern](https://charlesreid1.github.io/python-patterns-the-registry.html) -- Charles Reid
- [__init_subclass__ for class registries](https://blog.yuo.be/2018/08/16/__init_subclass__-a-simpler-way-to-implement-class-registries-in-python/) -- Reupen's blog
- [Typing a class decorator registry pattern](https://github.com/python/typing/discussions/1565) -- python/typing Discussion
- [PEP 443: Single-dispatch generic functions](https://peps.python.org/pep-0443/)
- [Starlette routing.py source](https://github.com/encode/starlette/blob/master/starlette/routing.py)
- [Dictionary dispatch pattern (HN discussion)](https://news.ycombinator.com/item?id=37271162)
- [Sudoblark: Python dictionary dispatch pattern](https://sudoblark.com/2024/03/25/the-python-dictionary-dispatch-pattern/)
