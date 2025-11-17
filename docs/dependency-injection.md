# Dependency Injection Patterns

## FastMCP: Closure-Based Registration

**Pattern:** Initialize dependencies → Create service → Register via closures

### Why Closures?

FastMCP has no `Depends()` system. Closures are the required pattern for accessing application-scoped state.

### The Pattern

```python
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
import tempfile
from pathlib import Path

class ServerState:
    """Server state with guaranteed initialization."""

    @classmethod
    async def create(cls):
        """Factory - fails fast if initialization fails."""
        temp_dir = tempfile.TemporaryDirectory()
        return cls(temp_dir, Path(temp_dir.name))

    def __init__(self, temp_dir: tempfile.TemporaryDirectory, output_dir: Path):
        self.temp_dir = temp_dir
        self.output_dir = output_dir


class PythonInterpreterService:
    """Service with non-Optional dependencies."""

    def __init__(self, state: ServerState):
        self.state = state  # NOT Optional - guaranteed to exist

    async def execute(self, code: str) -> dict:
        # self.state is guaranteed non-None
        output_file = self.state.output_dir / "output.txt"
        result = await run_code(code)
        output_file.write_text(result)
        return {"success": True, "output": result}


def register_tools(service: PythonInterpreterService):
    """Register service methods as tools via closures."""

    @mcp.tool()
    async def execute_code(code: str, ctx: Context) -> dict:
        # Closure captures 'service' from enclosing scope
        await ctx.info("Executing code...")
        return await service.execute(code)


@asynccontextmanager
async def lifespan(server):
    # Initialize state - fails if creation fails
    state = await ServerState.create()

    # Create service with non-Optional state
    service = PythonInterpreterService(state)

    # Register tools - closures capture service instance
    register_tools(service)

    yield

    # Cleanup
    state.temp_dir.cleanup()


mcp = FastMCP("python-interpreter", lifespan=lifespan)
```

### Benefits

- ✅ Type safety: `service.state` is `ServerState`, not `ServerState | None`
- ✅ Fail-fast: Initialization failure prevents server startup
- ✅ No assertions needed: Type checker knows state exists
- ✅ Testable: Instantiate service directly with mock state

### Testing Pattern

Services with explicit dependencies are trivially testable:

```python
# tests/conftest.py
import pytest

@pytest.fixture
async def server_state():
    """Create test state without server machinery."""
    state = await ServerState.create()
    yield state
    state.temp_dir.cleanup()

@pytest.fixture
def interpreter_service(server_state):
    """Create isolated service instance."""
    return PythonInterpreterService(server_state)


# tests/test_interpreter.py
async def test_execute(interpreter_service):
    """Test service directly - no server needed."""
    result = await interpreter_service.execute("print('hello')")
    assert result["success"] is True
    assert "hello" in result["output"]
```

## FastAPI: Depends() for Request-Scoped

FastAPI provides `Depends()` for request-scoped dependencies. Use it.

### Pattern

```python
from fastapi import FastAPI, Depends, Request

def get_user_service(request: Request) -> UserService:
    """Retrieve service from app.state."""
    return request.app.state.user_service

@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service)
) -> User:
    return await service.get_user(user_id)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize application-scoped services
    db = await DatabaseState.create(settings.DATABASE_URL)
    user_service = UserService(db)

    # Store on app.state
    app.state.user_service = user_service

    yield

    await db.close()

app = FastAPI(lifespan=lifespan)
```

### When to Use Each

- **Closures (FastMCP)**: Application-scoped state (required by framework)
- **Depends() (FastAPI)**: Request-scoped dependencies (idiomatic, recommended)

## Summary

**Initialize non-Optional dependencies at startup, register via closures.**

Type safety, fail-fast, testability, explicit dependencies.