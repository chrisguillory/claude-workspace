---
title: "Decomposing BrowserService: Patterns for Large Python Service Classes"
date: 2026-04-05
scope: selenium-browser-automation MCP server
---

# Decomposing BrowserService: Patterns for Large Python Service Classes

## 1. The Problem

The `selenium-browser-automation` MCP server's `server.py` is 4,334 lines. Its
architecture centers on three components:

1. **`BrowserState`** (lines 224-277): Mutable container holding WebDriver, temp
   directories, proxy config, origin tracking, storage caches, profile state, and
   response body capture flags. 14 mutable attributes.

2. **`BrowserService`** (lines 280-395): Thin wrapper around `BrowserState` with
   two methods: `close_browser()` and `get_browser()`. Every tool needs
   `service.get_browser()` to obtain the WebDriver.

3. **`register_tools(service)`** (lines 398-3712): A single function containing 31
   tool closures plus 2 inner helper functions, all capturing `service` from the
   enclosing scope. This is where the actual logic lives.

Additionally, there are 9 private module-level helper functions (lines 3765-4298)
that operate on `BrowserService` and/or `webdriver.Chrome`.

### Current Method Inventory

**31 tool closures in `register_tools()`:**

| Category | Tools | Lines (approx) |
|----------|-------|-----------------|
| Navigation | `navigate`, `navigate_with_profile_state` | ~420 |
| Page extraction | `get_page_text`, `get_page_html`, `get_aria_snapshot`, `get_visual_tree` | ~500 |
| Interaction | `click`, `hover`, `press_key`, `type_text`, `scroll` | ~430 |
| Waiting | `wait_for_network_idle`, `wait_for_selector`, `sleep` | ~300 |
| Elements | `get_interactive_elements`, `get_focusable_elements` | ~180 |
| Screenshots | `screenshot` | ~50 |
| JavaScript | `execute_javascript` | ~130 |
| Performance | `capture_web_vitals`, `get_resource_timings`, `export_har` | ~420 |
| Console | `get_console_logs` | ~100 |
| Profile state | `save_profile_state`, `export_chrome_profile_state`, `navigate_with_profile_state` | ~450 |
| Proxy | `configure_proxy`, `clear_proxy` | ~130 |
| Network | `set_blocked_urls`, `download_resource` | ~80 |
| Browser mgmt | `list_chrome_profiles`, `resize_window` | ~50 |

**2 inner helpers in `register_tools()`:**
- `_download_with_browser_context` (lines 1114-1158)
- `_lookup_intercepted_body` (lines 2549-2616)

**9 module-level helpers:**
- `_validate_css_selector`
- `_count_tree_nodes`
- `_build_page_metadata`
- `_save_large_output_to_file`
- `_cdp_enable_domstorage`, `_cdp_get_storage`, `_cdp_set_storage`
- `_capture_current_origin_storage`
- `_restore_pending_profile_state_for_current_origin`
- `_load_profile_state_from_file`, `_build_storage_init_script`
- `_inject_cookies_via_cdp`
- `_setup_pending_profile_state`
- `_install_response_body_capture_if_needed`
- `_sync_cleanup`

### Dependency Graph

Every tool closure follows the same pattern:

```
tool closure  -->  service.get_browser()  -->  driver: webdriver.Chrome
              -->  service.state.*         -->  BrowserState fields
              -->  _helper_function(service, driver)
```

The shared state access breaks down into these clusters:

**Cluster A: Navigation + Storage Management**
- `navigate`, `navigate_with_profile_state`, `click`, `press_key`
- These all call `_capture_current_origin_storage` before acting and
  `_restore_pending_profile_state_for_current_origin` after.
- Access: `state.origin_tracker`, `state.local_storage_cache`,
  `state.session_storage_cache`, `state.indexed_db_cache`,
  `state.pending_profile_state`, `state.restored_origins`

**Cluster B: Page Extraction**
- `get_page_text`, `get_page_html`, `get_aria_snapshot`, `get_visual_tree`
- Read-only: only need `driver` and `state.screenshot_dir`

**Cluster C: Interaction**
- `click`, `hover`, `press_key`, `type_text`, `scroll`
- Need `driver`. `click` and `press_key` also touch Cluster A state.

**Cluster D: Profile State**
- `save_profile_state`, `export_chrome_profile_state`, `navigate_with_profile_state`
- Heavy access: `state.origin_tracker`, all storage caches, `state.pending_profile_state`

**Cluster E: Performance / Network**
- `capture_web_vitals`, `get_resource_timings`, `export_har`, `get_console_logs`
- Need `driver` and `state.capture_dir`, `state.response_body_capture_enabled`

**Cluster F: Proxy**
- `configure_proxy`, `clear_proxy`
- Access: `state.proxy_config`, `state.mitmproxy_process`

**Cluster G: Browser Management**
- `list_chrome_profiles`, `resize_window`, `set_blocked_urls`, `download_resource`, `sleep`
- Mostly just need `driver`; `download_resource` also reads `state.proxy_config`

### The Binding Constraint

All 31 tools must be registered on a single `FastMCP` instance via the `@mcp.tool()`
decorator. The current architecture uses closure capture:

```python
def register_tools(service: BrowserService) -> None:
    @mcp.tool()
    async def navigate(url: str, ...) -> NavigationResult:
        driver = await service.get_browser()
        ...
```

Any decomposition must preserve this registration mechanism or provide an equivalent.


---

## 2. Pattern Research

### 2.1 Mixin Classes

**How it works:** Split methods into multiple base classes, each providing a subset
of behavior. The final class inherits from all mixins.

**Real-world examples:**
- Django class-based views decompose into `LoginRequiredMixin`, `PermissionRequiredMixin`,
  `SingleObjectMixin`, `MultipleObjectMixin`, etc.
  ([Django docs](https://docs.djangoproject.com/en/6.0/topics/class-based-views/mixins/))
- Django REST Framework's `GenericAPIView` composes `CreateModelMixin`,
  `ListModelMixin`, `RetrieveModelMixin`, `UpdateModelMixin`, `DestroyModelMixin`.
- ThreadingMixIn / ForkingMixIn in Python's `socketserver` module.

**MRO considerations:**
Python uses C3 linearization. Mixins must come before base classes in the inheritance
list. Each mixin should call `super().__init__()` to preserve the chain.
([Python MRO docs](https://www.python.org/download/releases/2.3/mro/),
[Real Python](https://realpython.com/python-mixin/))

**Type-checking with mypy:**
Adding type hints to mixin classes is tricky because they implicitly depend on
attributes from the host class. Adam Johnson's 2025 article
([adamj.eu](https://adamj.eu/tech/2025/05/01/python-type-hints-mixin-classes/))
recommends two approaches:
1. Declare the mixin as inheriting from the required base class.
2. Use a `Protocol` as a self-type annotation for looser coupling.

**Key limitation for BrowserService:** Django mixins work well because each mixin
is *stateless* -- it adds behavior, not data. The Django design guidance explicitly
says: "Mixins generally work best when they don't maintain significant state."
([Leapcell](https://leapcell.io/blog/understanding-django-mixins-a-deep-dive-into-loginrequiredmixin-and-custom-implementations))
BrowserService's tools all share heavy mutable state via `BrowserState`.

### 2.2 Composition / Delegation

**How it works:** BrowserService holds references to sub-service objects, each
responsible for a cluster of operations. Sub-services receive `BrowserState` (or a
reference to `BrowserService`) in their constructor.

**Real-world examples:**
- Selenium's own `WebDriver` class (1,593 lines, 1,276 loc) uses composition:
  it delegates to `SwitchTo`, `Mobile`, and `FedCM` helper objects rather than
  using inheritance.
  ([GitHub](https://github.com/SeleniumHQ/selenium/blob/trunk/py/selenium/webdriver/remote/webdriver.py))
- Playwright's architecture decomposes into `Browser` > `BrowserContext` > `Page`,
  with each layer owning a specific scope of state.
  ([DeepWiki](https://deepwiki.com/microsoft/playwright-python/3.2-browsers-and-contexts))
- Python `requests.Session` delegates to adapters via `mount()`.
  ([source](https://requests.readthedocs.io/en/latest/_modules/requests/sessions/))
- The fast.ai `GetAttr` base class automates delegation via `__getattr__`.
  ([fast.ai](https://www.fast.ai/posts/2019-08-06-delegation.html))

**Shared state handling:** The sub-service receives a reference to the shared state
object. This is explicit dependency injection -- the sub-service constructor takes
`state: BrowserState`. All sub-services point to the same `BrowserState` instance.

**Trade-off:** IDE autocomplete and type checking work well because the sub-service's
interface is explicit. But you need forwarding methods on the facade if you want
`service.navigate()` to still work (or you use `getattr` dispatch).

### 2.3 Protocol Classes / Structural Subtyping

**How it works:** Define `Protocol` classes that describe subsets of the service's
interface. Consumer code depends on the Protocol, not the concrete class.
([PEP 544](https://peps.python.org/pep-0544/))

**PEP 544 recommendation:** "It is recommended to create compact protocols and
combine them" rather than creating large protocols. This aligns with the Interface
Segregation Principle.
([mypy docs](https://mypy.readthedocs.io/en/stable/protocols.html))

**Assessment:** Protocols help *consumers* depend on narrow interfaces, but they
do not help *decompose the implementation* across files. A class can satisfy multiple
Protocols without any structural change. This is a type-checking tool, not an
organization tool.

### 2.4 Module-Level Functions (No Class)

**How it works:** Eliminate the service class entirely. Each tool is an async function
that takes `state: BrowserState` as its first argument.

**Real-world examples:**
- Django service layers often use plain functions.
  ([hashnode](https://simoncrowe.hashnode.dev/django-service-layers-beyond-fat-models-vs-enterprise-patterns))
- The "Cosmic Python" architecture book recommends stateless service functions.
  ([cosmicpython.com](https://www.cosmicpython.com/book/chapter_04_service_layer.html))
- Bas Nijholt's argument for functional Python: "Functions over classes? Why I
  prefer a simpler, functional style."
  ([nijho.lt](https://www.nijho.lt/post/functional-python/))

**Key insight:** Service classes that hold no state of their own (BrowserService only
holds `self.state`) are effectively namespaces around functions. The class adds no
value beyond grouping.

**Assessment:** This is the closest to the *current* architecture -- `register_tools`
already uses closures (which are effectively module-level functions capturing state).
The question is whether to keep the closure pattern or convert to explicit function
signatures with `state` parameters.

### 2.5 `__init_subclass__` / Class Registries

**How it works:** Use `__init_subclass__` to automatically register methods from
subclasses defined in separate files.
([PEP 487](https://peps.python.org/pep-0487/),
[zetcode](https://zetcode.com/python/dunder-init_subclass/))

**Assessment:** This pattern is designed for *plugin discovery* -- finding all
implementations of an interface. BrowserService's methods are not interchangeable
plugins; they are a fixed set of browser operations. The pattern adds complexity
without addressing the core problem of file size.

### 2.6 Splitting a Class Across Multiple Files

Mark Summerfield documents four approaches
([mark-summerfield.github.io](https://mark-summerfield.github.io/pyclassmulti.html)):

1. **Delegation to functions:** Methods call standalone functions, passing `self`.
2. **Decorator with function registration:** `@register_function` collects methods
   from separate files; a class decorator attaches them.
3. **Refined decorator approach:** Uses `__methods__` lists per module.
4. **Mixin inheritance:** The "definitive" approach -- each file defines a Mixin
   class with a subset of methods; the main class inherits from all Mixins.

Summerfield concludes that mixin inheritance is simplest because it requires no
library code and "just works" with standard Python.

### 2.7 Extension Methods Pattern

**How it works:** Define functions outside the class, attach them at runtime via
`setattr()` or decorators.

**Assessment:** This breaks static analysis, IDE support, and type checking. Mypy
cannot verify methods that are attached dynamically. The "Snake Eyes" article
([dev.to](https://dev.to/tmr232/snake-eyes-extension-methods-1jb)) explores
Python equivalents to Kotlin extension functions but concludes they are limited.

### 2.8 C# Partial Classes

**How it works:** C# allows `partial class Foo` in multiple files. The compiler
merges them into a single class.

**Python equivalent:** Python has no language-level support for partial classes.
The closest approximations are mixins or decorator-based method registration
(Summerfield's approach above).

### 2.9 Rust impl Blocks

**How it works:** Rust allows multiple `impl Struct` blocks, potentially in different
files within the same module.
([Rust forum](https://users.rust-lang.org/t/code-structure-for-big-impl-s-distributed-over-several-files/7785))

**Python equivalent:** The mixin pattern is the closest analog. Python cannot
natively add methods to an existing class from another file without runtime tricks.

### 2.10 Go Struct Embedding

**How it works:** Go embeds one struct in another, promoting its methods. Interface
satisfaction is implicit (structural typing).
([Go by Example](https://gobyexample.com/struct-embedding))

**Python equivalent:** Composition with `__getattr__` forwarding (fast.ai pattern)
or explicit delegation. Python's `Protocol` class provides the structural typing
aspect.

### 2.11 Ruby Modules/Mixins

**How it works:** Ruby's `include ModuleName` adds a module's methods to a class.
Modules are not classes and cannot be instantiated.
([Andrew Brookins](https://andrewbrookins.com/technology/mixins-in-python-and-ruby-compared/))

**Python equivalent:** Python mixins via multiple inheritance. The key difference
is that Ruby modules are a distinct construct, while Python uses regular classes
as mixins by convention.

### 2.12 Command Pattern

**How it works:** Each operation becomes a Command object with an `execute()` method.
([refactoring.guru](https://refactoring.guru/design-patterns/command/python/example))

**Assessment for BrowserService:** Each tool would become a class with state access:

```python
class NavigateCommand:
    def __init__(self, state: BrowserState):
        self.state = state
    async def execute(self, url: str, ...) -> NavigationResult:
        ...
```

This adds significant indirection for no clear benefit. Browser tools are not
undoable, queueable, or serializable -- the use cases that justify the Command
pattern. It would also break the `@mcp.tool()` decorator registration.

### 2.13 Facade Pattern

**How it works:** A thin interface that delegates to multiple internal services.
([refactoring.guru](https://refactoring.guru/design-patterns/facade/python/example))

**Assessment:** This is essentially composition viewed from the consumer's
perspective. BrowserService *is already* the facade -- it is the public API. The
question is what sits behind it.

### 2.14 Mediator Pattern

**How it works:** A central coordinator dispatches to specialized handlers. Modern
Python implementations like `mediatr` use type-based dispatch.
([refactoring.guru](https://refactoring.guru/design-patterns/mediator/python/example))

**Assessment:** Overkill for this case. The tools are registered at startup, not
dynamically dispatched. The MCP framework already acts as the mediator.

### 2.15 How Large Projects Handle This

**Selenium WebDriver** (1,593 lines): Monolithic class. Uses composition for
`SwitchTo`, `Mobile`, `FedCM` but keeps 40+ methods directly on `WebDriver`.
No mixins, no inheritance chain beyond `BaseWebDriver`.

**Playwright Python**: Decomposes by *lifecycle scope* (Browser > BrowserContext >
Page), not by method type. Each level is a separate class with its own state.
Communication is via the Channel/Connection protocol layer.

**requests.Session** (~750 lines): Monolithic class inheriting `SessionRedirectMixin`.
Uses composition for transport adapters. Relatively small -- 15-20 public methods.

**SQLAlchemy Session** (~2,000+ lines): Monolithic class. Complex internal state
management with `IdentityMap` and `SessionTransaction` as composed objects, but
the Session class itself remains large.

**Django QuerySet** (~1,800 lines): Single file `django/db/models/query.py`. Not
decomposed via mixins despite Django's heavy use of mixins for views.

**The pattern is clear:** Large Python service classes in mature projects tend to
remain monolithic. Selenium, SQLAlchemy, and Django all keep their core state-heavy
classes in single files. Decomposition happens at *architectural boundaries* (e.g.,
Playwright's Browser/Context/Page), not at the *method grouping* level.


---

## 3. Applying Patterns to BrowserService

### Constraint: The MCP Tool Registration

The binding constraint is how tools are registered:

```python
@mcp.tool()
async def navigate(url: str, ...) -> NavigationResult:
    driver = await service.get_browser()
    ...
```

Any pattern must produce functions or methods that can be decorated with
`@mcp.tool()`. The decorator expects a standalone async function with typed
parameters -- it inspects the function's signature to generate the MCP tool schema.

### Pattern 1: Mixin-Based BrowserService (File per Cluster)

```
selenium_browser_automation/
    service/
        __init__.py          # Re-exports BrowserService
        _state.py            # BrowserState, OriginTracker
        _base.py             # BrowserService(NavigationMixin, ExtractionMixin, ...)
        _navigation.py       # NavigationMixin
        _extraction.py       # ExtractionMixin
        _interaction.py      # InteractionMixin
        _waiting.py          # WaitingMixin
        _performance.py      # PerformanceMixin
        _profile_state.py    # ProfileStateMixin
        _proxy.py            # ProxyMixin
        _helpers.py          # Module-level helpers
    server.py               # FastMCP setup, register_tools()
```

**Implementation:**

```python
# _state.py
class BrowserState:
    """Container for all browser state."""
    ...  # Unchanged from current

# _navigation.py
from ._state import BrowserState

class NavigationMixin:
    """Navigation tools: navigate, navigate_with_profile_state."""

    state: BrowserState  # Typed for mypy via Protocol or base class

    async def navigate(self, url: str, fresh_browser: bool = False, ...) -> NavigationResult:
        driver = await self.get_browser()
        await _capture_current_origin_storage(self, driver)
        await asyncio.to_thread(driver.get, url)
        ...

    async def navigate_with_profile_state(self, url: str, ...) -> NavigationResult:
        ...

# _base.py
class BrowserService(
    NavigationMixin,
    ExtractionMixin,
    InteractionMixin,
    WaitingMixin,
    PerformanceMixin,
    ProfileStateMixin,
    ProxyMixin,
):
    """Browser automation service - protocol-agnostic, pure domain logic."""

    def __init__(self, state: BrowserState) -> None:
        self.state = state

    async def close_browser(self) -> None: ...
    async def get_browser(self, ...) -> webdriver.Chrome: ...
```

**register_tools() changes:**

```python
def register_tools(service: BrowserService) -> None:
    @mcp.tool()
    async def navigate(url: str, ...) -> NavigationResult:
        return await service.navigate(url, ...)
    # Each closure becomes a thin delegation to a service method
```

**Assessment:**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| File size reduction | Good | ~7 files of 200-500 lines each |
| Type safety (mypy) | Moderate | Mixins need `state: BrowserState` annotation; Adam Johnson's Protocol approach works but error messages are cryptic |
| `getattr` dispatch | Yes | Methods are on the BrowserService instance |
| Shared state access | Yes | All mixins access `self.state` |
| File count | ~10 | service/ package with 8-9 files |
| Cognitive overhead | Moderate | Must check MRO to find method; IDE "go to definition" works |
| Preserves tool registration | Yes | register_tools delegates to service methods |

**Problems:**
- Mixins share *all* of BrowserState even when they only need a subset. A
  NavigationMixin can read `proxy_config` even though it has no reason to.
- No encapsulation of state access -- any mixin can mutate any field.
- Mixin ordering in the class definition matters for MRO; adding a new mixin
  can subtly change method resolution.
- The `register_tools()` function remains large (31 thin delegation closures),
  though each is now one-liners. Alternatively, each mixin could register its
  own tools, but that couples the mixin to the FastMCP instance.

### Pattern 2: Composition with Sub-Services

```
selenium_browser_automation/
    service/
        __init__.py          # Re-exports BrowserService
        _state.py            # BrowserState, OriginTracker
        _browser_service.py  # BrowserService (facade)
        _navigation.py       # NavigationService
        _extraction.py       # ExtractionService
        _interaction.py      # InteractionService
        _waiting.py          # WaitingService
        _performance.py      # PerformanceService
        _profile_state.py    # ProfileStateService
        _proxy.py            # ProxyService
        _helpers.py          # Shared helpers
    server.py               # FastMCP setup, register_tools()
```

**Implementation:**

```python
# _navigation.py
class NavigationService:
    """Navigation operations on a shared browser session."""

    def __init__(self, state: BrowserState, get_browser: Callable, close_browser: Callable) -> None:
        self._state = state
        self._get_browser = get_browser
        self._close_browser = close_browser

    async def navigate(self, url: str, fresh_browser: bool = False, ...) -> NavigationResult:
        if fresh_browser:
            await self._close_browser()
        driver = await self._get_browser(...)
        await _capture_current_origin_storage_from_state(self._state, driver)
        ...

# _browser_service.py
class BrowserService:
    """Facade: composes sub-services, owns browser lifecycle."""

    def __init__(self, state: BrowserState) -> None:
        self.state = state
        self.navigation = NavigationService(state, self.get_browser, self.close_browser)
        self.extraction = ExtractionService(state, self.get_browser)
        self.interaction = InteractionService(state, self.get_browser)
        self.waiting = WaitingService(state, self.get_browser)
        self.performance = PerformanceService(state, self.get_browser)
        self.profile_state = ProfileStateService(state, self.get_browser, self.close_browser)
        self.proxy = ProxyService(state, self.close_browser)

    async def close_browser(self) -> None: ...
    async def get_browser(self, ...) -> webdriver.Chrome: ...
```

**register_tools() changes:**

```python
def register_tools(service: BrowserService) -> None:
    @mcp.tool()
    async def navigate(url: str, ...) -> NavigationResult:
        return await service.navigation.navigate(url, ...)

    @mcp.tool()
    async def get_page_text(selector: str = 'auto', ...) -> PageTextResult:
        return await service.extraction.get_page_text(selector, ...)
```

**Assessment:**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| File size reduction | Good | Same as mixins (~7 files of 200-500 lines) |
| Type safety (mypy) | Excellent | Each sub-service has explicit typed constructor |
| `getattr` dispatch | Partial | Need `service.navigation.navigate` not `service.navigate` |
| Shared state access | Yes | All sub-services hold reference to same `BrowserState` |
| File count | ~10 | Same as mixins |
| Cognitive overhead | Low-Moderate | Clear ownership; no MRO complexity |
| Preserves tool registration | Yes | register_tools delegates through sub-services |

**Problems:**
- Constructor injection of `get_browser` / `close_browser` callables is awkward.
  Alternatively, sub-services could take the BrowserService itself, but that
  creates a circular reference.
- `register_tools()` still has 31 closures (thin delegations).
- Navigation and profile state services both need `close_browser()`, creating
  cross-cutting dependency.

### Pattern 3: Module-Level Functions (Current Architecture, Cleaned Up)

```
selenium_browser_automation/
    tools/
        __init__.py          # Re-exports register_all_tools
        _navigation.py       # register_navigation_tools(mcp, service)
        _extraction.py       # register_extraction_tools(mcp, service)
        _interaction.py      # register_interaction_tools(mcp, service)
        _waiting.py          # register_waiting_tools(mcp, service)
        _performance.py      # register_performance_tools(mcp, service)
        _profile_state.py    # register_profile_state_tools(mcp, service)
        _proxy.py            # register_proxy_tools(mcp, service)
        _browser_mgmt.py     # register_browser_mgmt_tools(mcp, service)
    _state.py                # BrowserState, OriginTracker
    _service.py              # BrowserService (get_browser, close_browser)
    _helpers.py              # Module-level helpers
    server.py                # FastMCP setup, lifespan, main
```

**Implementation:**

```python
# tools/_navigation.py
from .._service import BrowserService
from .._helpers import _capture_current_origin_storage, _restore_pending_profile_state_for_current_origin

def register_navigation_tools(mcp: FastMCP, service: BrowserService) -> None:
    """Register navigation tools on the MCP server."""

    @mcp.tool(annotations=ToolAnnotations(title='Navigate to URL', ...))
    async def navigate(
        url: str,
        fresh_browser: bool = False,
        enable_har_capture: bool = False,
        init_scripts: Sequence[str] | None = None,
        browser: Browser | None = None,
        ctx: Context | None = None,
    ) -> NavigationResult:
        """Load a URL and establish browser session. ..."""
        # Exact same implementation as today
        if fresh_browser:
            await service.close_browser()
        driver = await service.get_browser(enable_har_capture=enable_har_capture, browser=browser)
        await _capture_current_origin_storage(service, driver)
        await asyncio.to_thread(driver.get, url)
        ...

# server.py
from .tools import register_all_tools

@contextlib.asynccontextmanager
async def lifespan(server_instance: FastMCP):
    state = await BrowserState.create()
    service = BrowserService(state)
    register_all_tools(mcp, service)
    ...

# tools/__init__.py
def register_all_tools(mcp: FastMCP, service: BrowserService) -> None:
    register_navigation_tools(mcp, service)
    register_extraction_tools(mcp, service)
    register_interaction_tools(mcp, service)
    register_waiting_tools(mcp, service)
    register_performance_tools(mcp, service)
    register_profile_state_tools(mcp, service)
    register_proxy_tools(mcp, service)
    register_browser_mgmt_tools(mcp, service)
```

**Assessment:**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| File size reduction | Excellent | 8 tool files of 100-500 lines; server.py drops to ~150 lines |
| Type safety (mypy) | Excellent | No mixin complexity; functions capture typed `service` |
| `getattr` dispatch | N/A | Tools are registered via decorator, not dispatched via getattr |
| Shared state access | Yes | All closures capture same `service` reference |
| File count | ~12 | tools/ package with 8 files + state + service + helpers + server |
| Cognitive overhead | Low | Each file is self-contained; no inheritance to trace |
| Preserves tool registration | Perfect | Identical pattern to current code, just in separate files |

**This is the natural evolution of the current architecture.** It requires no
structural changes -- only splitting the single `register_tools()` function into
multiple `register_X_tools()` functions in separate files.

### Pattern 4: Hybrid Mixin + Separate Tool Registration

```
selenium_browser_automation/
    service/
        __init__.py
        _state.py
        _base.py             # BrowserService base (get_browser, close_browser)
        _navigation.py       # NavigationMixin + register_navigation_tools()
        _extraction.py       # ExtractionMixin + register_extraction_tools()
        ...
    server.py
```

Each file defines both a mixin class (the implementation) and a registration
function (the MCP binding):

```python
# _navigation.py
class NavigationMixin:
    async def _navigate_impl(self, url: str, ...) -> NavigationResult:
        driver = await self.get_browser()
        ...

def register_navigation_tools(mcp: FastMCP, service: 'BrowserService') -> None:
    @mcp.tool()
    async def navigate(url: str, ...) -> NavigationResult:
        return await service._navigate_impl(url, ...)
```

**Assessment:** This combines the overhead of both patterns (mixin complexity +
separate registration) without clear advantages over either alone. The mixin's
only purpose is to provide `self` access, which the closure already handles.

### Pattern 5: Command Objects

```python
class NavigateCommand:
    def __init__(self, service: BrowserService) -> None:
        self.service = service

    async def __call__(self, url: str, fresh_browser: bool = False, ...) -> NavigationResult:
        if fresh_browser:
            await self.service.close_browser()
        driver = await self.service.get_browser()
        ...

# Registration
navigate_cmd = NavigateCommand(service)
mcp.add_tool(navigate_cmd, name="navigate", ...)
```

**Assessment:**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| File size reduction | Good | One class per file is natural |
| Type safety | Good | Explicit typed constructor |
| Cognitive overhead | High | 31 classes for 31 tools is excessive ceremony |
| Preserves tool registration | Uncertain | FastMCP's `@mcp.tool()` decorator inspects function signatures; `__call__` may not work without adaptation |

The Command pattern adds a class-per-tool without enabling undo, queuing, or
serialization -- the features that justify it.


---

## 4. Comparative Analysis

| Criterion | Mixins | Composition | Split register_tools | Hybrid | Command |
|-----------|--------|-------------|----------------------|--------|---------|
| Migration effort | High | High | **Low** | Very High | High |
| Breaking changes | Some | Some | **None** | Some | Many |
| Type safety | Moderate | Good | **Excellent** | Moderate | Good |
| IDE support | Good | Good | **Excellent** | Good | Poor |
| File size per file | 200-500 | 200-500 | **100-500** | 200-500 | 80-150 |
| Cognitive overhead | Moderate | Low-Moderate | **Low** | High | High |
| Preserves current arch | Partial | No | **Yes** | Partial | No |
| Cross-cutting concerns | Hard | Hard | **Easy** | Hard | Hard |


---

## 5. Recommendation: Split register_tools (Pattern 3)

### Why This Pattern

1. **Minimal migration effort.** Every tool closure is *already* a standalone
   function. Moving it to a separate file requires only adding imports and changing
   the enclosing function name. No refactoring of BrowserService, BrowserState,
   or the helper functions is needed.

2. **Zero breaking changes.** The MCP tool registration mechanism, the FastMCP
   decorator, and the lifespan management all remain identical. The public API
   (tool names, parameters, return types) is unchanged.

3. **Excellent type safety.** Closures capturing a typed `service: BrowserService`
   parameter are fully understood by mypy. No Protocol tricks, no mixin self-type
   annotations, no dynamic method attachment.

4. **Natural cluster boundaries.** The dependency analysis in Section 1 identified
   7 clusters. Each becomes a file. The clusters were derived from which state
   fields each tool accesses, so the grouping reflects real cohesion.

5. **Incremental migration.** Tools can be moved one cluster at a time. At every
   step the server remains fully functional. No big-bang refactor required.

6. **Matches industry precedent.** Selenium, SQLAlchemy, and Django keep their
   state-heavy core classes monolithic but split *registration* or *configuration*
   across files. This is the same approach.

### Why Not Mixins

The python-patterns.guide article on composition over inheritance
([source](https://python-patterns.guide/gang-of-four/composition-over-inheritance/))
explicitly critiques mixins:

> "Mixins improve readability but retain multiple inheritance's liabilities
> regarding method resolution order and combination guarantees."

Django's mixin success relies on *stateless* mixins with *simple* method chains.
BrowserService's tools mutate heavy shared state (`BrowserState` has 14 mutable
attributes) and call shared helpers that also mutate state. This makes mixin
boundaries porous -- every mixin can reach into any state field, and MRO changes
can silently alter behavior.

### Why Not Composition

Composition via sub-services is architecturally cleaner than mixins but introduces
unnecessary indirection for this case. The current closures already *are* the
implementation -- wrapping them in sub-service classes adds a layer of objects
without adding capability. It would make sense if BrowserService needed to be
assembled from *interchangeable* components (e.g., different navigation strategies),
but the implementation is fixed.

Additionally, cross-cutting concerns like `_capture_current_origin_storage` (called
by navigate, click, and press_key) would need to be shared between NavigationService
and InteractionService, either via shared helper functions (which is what Pattern 3
already does) or via a common base class (which reintroduces inheritance).

### Proposed File Layout

```
selenium_browser_automation/
    __init__.py
    server.py                    # ~150 lines: FastMCP setup, lifespan, main, mcp instance
    state.py                     # ~120 lines: BrowserState, OriginTracker
    service.py                   # ~120 lines: BrowserService (get_browser, close_browser)
    helpers.py                   # ~550 lines: All private helper functions
    tools/
        __init__.py              # ~40 lines: register_all_tools()
        navigation.py            # ~420 lines: navigate, navigate_with_profile_state
        extraction.py            # ~350 lines: get_page_text, get_page_html, get_aria_snapshot, get_visual_tree
        interaction.py           # ~430 lines: click, hover, press_key, type_text, scroll
        waiting.py               # ~300 lines: wait_for_network_idle, wait_for_selector, sleep
        performance.py           # ~420 lines: capture_web_vitals, get_resource_timings, export_har, get_console_logs
        profile_state.py         # ~450 lines: save_profile_state, export_chrome_profile_state
        proxy.py                 # ~130 lines: configure_proxy, clear_proxy
        browser_mgmt.py          # ~130 lines: list_chrome_profiles, resize_window, set_blocked_urls, download_resource, execute_javascript
    models.py                    # Unchanged
    scripts/                     # Unchanged
    validators.py                # Unchanged
    scroll.py                    # Unchanged
    tree_utils.py                # Unchanged
    chrome_profiles.py           # Unchanged
    chrome_profile_state_export.py  # Unchanged
    ...
```

**Total: server.py drops from 4,334 lines to ~150 lines.** No single file exceeds
~550 lines.

### Migration Path

**Phase 1: Extract state and service (1 commit)**
- Move `BrowserState` and `OriginTracker` to `state.py`.
- Move `BrowserService` to `service.py`.
- Update imports in `server.py`.
- Run tests.

**Phase 2: Extract helpers (1 commit)**
- Move all `_private` helper functions to `helpers.py`.
- Update imports.
- Run tests.

**Phase 3: Extract tool clusters (1 commit per cluster, or 1 commit for all)**
- Create `tools/` package.
- Move each cluster's closures into its own `register_X_tools()` function.
- Create `tools/__init__.py` with `register_all_tools()`.
- Replace the monolithic `register_tools()` in `server.py` with a call to
  `register_all_tools()`.
- Run tests after each cluster.

**Phase 4: Clean up (1 commit)**
- Remove the now-empty `register_tools()` function.
- Update `__all__` exports.
- Verify all MCP tool registrations.

Each phase is independently shippable. Phase 3 can be done one cluster at a time
if a more conservative approach is desired.

### How This Affects the HTTP Bridge

If an HTTP bridge dispatches tools via `getattr(service, tool_name)`, this pattern
does *not* affect it -- the tools are registered on the `FastMCP` instance, not as
methods on `BrowserService`. The bridge calls `mcp.call_tool(tool_name, args)`,
which invokes the registered closure. The closure's location (which file it lives
in) is invisible to the bridge.

### When to Reconsider

Move to **composition** if:
- BrowserService needs to support multiple browser backends (not just Selenium).
- Sub-services need to be independently testable with mock state.
- A second consumer (e.g., a REST API) needs to reuse the service without MCP.

Move to **mixins** if:
- The service class itself (not just tool registration) grows beyond 500 lines.
- Methods need polymorphic dispatch (different behavior for different browser types).


---

## 6. Sources

### Python Patterns
- [What Are Mixin Classes in Python?](https://realpython.com/python-mixin/) -- Real Python
- [Composition Over Inheritance](https://python-patterns.guide/gang-of-four/composition-over-inheritance/) -- python-patterns.guide
- [Inheritance and Composition: A Python OOP Guide](https://realpython.com/inheritance-composition-python/) -- Real Python
- [Make Delegation Work in Python](https://www.fast.ai/posts/2019-08-06-delegation.html) -- fast.ai
- [Python Type Hints: Mixin Classes](https://adamj.eu/tech/2025/05/01/python-type-hints-mixin-classes/) -- Adam Johnson
- [Spreading a Class Over Multiple Files](https://mark-summerfield.github.io/pyclassmulti.html) -- Mark Summerfield
- [Method Delegation in Python](https://michaelcho.me/article/method-delegation-in-python/) -- Michael Cho

### Refactoring
- [Refactoring: This Class is Too Large](https://martinfowler.com/articles/class-too-large.html) -- Martin Fowler / Clare Sudbery
- [Large Class](https://refactoring.guru/smells/large-class) -- refactoring.guru
- [Tips for Refactoring a Mega Class](https://pybit.es/articles/tips-for-refactoring-a-mega-class/) -- Pybites
- [Refactoring the God Class in Python](https://betterprogramming.pub/refactoring-the-god-class-in-python-5c13942d0e75) -- Better Programming
- [God Class: The Definitive Guide](https://www.metridev.com/metrics/god-class-the-definitive-guide-to-identifying-and-avoiding-it/) -- Metridev

### Design Patterns
- [Command Pattern in Python](https://refactoring.guru/design-patterns/command/python/example) -- refactoring.guru
- [Facade Pattern in Python](https://refactoring.guru/design-patterns/facade/python/example) -- refactoring.guru
- [Mediator Pattern in Python](https://refactoring.guru/design-patterns/mediator/python/example) -- refactoring.guru
- [PEP 544: Protocols -- Structural Subtyping](https://peps.python.org/pep-0544/) -- Python.org
- [PEP 487: Simpler Customisation of Class Creation](https://peps.python.org/pep-0487/) -- Python.org

### Language-Specific Patterns
- [Django Mixins](https://docs.djangoproject.com/en/6.0/topics/class-based-views/mixins/) -- Django Documentation
- [Understanding Django Mixins](https://leapcell.io/blog/understanding-django-mixins-a-deep-dive-into-loginrequiredmixin-and-custom-implementations) -- Leapcell
- [Python MRO](https://www.python.org/download/releases/2.3/mro/) -- Python.org
- [Rust Impl Blocks Across Files](https://users.rust-lang.org/t/code-structure-for-big-impl-s-distributed-over-several-files/7785) -- Rust Forum
- [Go Struct Embedding](https://gobyexample.com/struct-embedding) -- Go by Example
- [Kotlin Extension Functions](https://kotlinlang.org/docs/extensions.html) -- Kotlin Documentation
- [Ruby Mixins vs Python](https://andrewbrookins.com/technology/mixins-in-python-and-ruby-compared/) -- Andrew Brookins

### Real-World Codebases
- [Selenium WebDriver source](https://github.com/SeleniumHQ/selenium/blob/trunk/py/selenium/webdriver/remote/webdriver.py) -- GitHub
- [Playwright Python Architecture](https://deepwiki.com/microsoft/playwright-python/3.2-browsers-and-contexts) -- DeepWiki
- [requests.Session source](https://requests.readthedocs.io/en/latest/_modules/requests/sessions/) -- Read the Docs
- [SQLAlchemy Session source](https://github.com/zzzeek/sqlalchemy/blob/main/lib/sqlalchemy/orm/session.py) -- GitHub

### Service Architecture
- [Django Service Layers](https://simoncrowe.hashnode.dev/django-service-layers-beyond-fat-models-vs-enterprise-patterns) -- Hashnode
- [Flask API and Service Layer](https://www.cosmicpython.com/book/chapter_04_service_layer.html) -- Cosmic Python
- [Functions Over Classes](https://www.nijho.lt/post/functional-python/) -- Bas Nijholt
