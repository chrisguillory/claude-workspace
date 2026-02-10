# exception_safety_linter.py: skip-file
# ruff: noqa: E722, F841, SIM105
# E722: bare except required for EXC001 test
# F841: unused exception variable required for EXC005 test
# SIM105: try-except-pass patterns demonstrate exception anti-patterns
"""Exception safety test cases - demonstrates violations and correct patterns.

Run the linter on this file: ./exception_safety_linter.py exception_safety_test_cases.py

Each rule has separate functions:
- VIOLATION: Code that triggers the linter (intentionally bad)
- CORRECT: Proper patterns to follow
- SUPPRESSED: Rare cases where suppression is justified (if any)

Organization:
- Public test functions (excNNN_*) demonstrate patterns
- Private helpers show supporting code

Design principle: "correct" examples should pass ALL linter rules, not just the
one they're demonstrating. If a pattern requires violating another rule, it
should use an explicit suppression directive with documentation.

Linter philosophy on nested handlers: When an outer exception handler re-raises,
nested handlers within it are assumed to handle cleanup/enrichment errors. The
linter won't flag broad catches in nested handlers because the outer re-raise
preserves the original exception. Using add_note() for enrichment is shown as
best practice but not enforced - production code often uses logging or silent
suppression for cleanup errors.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import traceback
import types
from collections.abc import Generator, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# EXC001: Bare except (catches KeyboardInterrupt, SystemExit, CancelledError, GeneratorExit)
# =============================================================================


def exc001_violation_basic() -> None:
    """VIOLATION: Bare except catches system exceptions."""
    try:
        risky_operation()
    except:  # EXC001: catches KeyboardInterrupt, SystemExit, CancelledError, GeneratorExit
        pass


def exc001_correct() -> None:
    """CORRECT: Replace bare except with explicit exception handling.

    NOTE: Bare except is bad because it catches system exceptions
    (KeyboardInterrupt, SystemExit, CancelledError, GeneratorExit).
    All patterns here either catch specific types OR re-raise.
    For intentional suppression, see exc002_suppressed_intentional().
    """
    # Option 1: Catch specific expected exceptions (PREFERRED)
    # This is the best approach when you know what can go wrong
    try:
        risky_operation()
    except (ValueError, TypeError):
        handle_error()

    # Option 2: Catch Exception + re-raise (error boundary pattern)
    # Use when you need cleanup before propagation
    try:
        risky_operation()
    except Exception:
        cleanup()
        raise  # Re-raise after cleanup

    # Option 3: Catch BaseException + re-raise (strongest error boundary)
    # Use when cleanup is needed even for KeyboardInterrupt/SystemExit
    try:
        risky_operation()
    except BaseException:
        cleanup()
        raise  # MUST re-raise - this is an error boundary, not suppression


# NOTE: No suppression example for EXC001. There is no valid use case for
# bare `except:` when `except BaseException:` exists and is explicit.
# If you truly need to catch everything, use `except BaseException:`.


# =============================================================================
# EXC002: Swallowed exception (broad catch without re-raise)
# =============================================================================


def exc002_violation_basic() -> None:
    """VIOLATION: Catching Exception and not re-raising hides errors."""
    try:
        risky_operation()
    except Exception:  # EXC002: broad catch without raise
        return None  # Swallows exception!


def exc002_violation_base_exception() -> None:
    """VIOLATION: BaseException without re-raise is even worse."""
    try:
        risky_operation()
    except BaseException:  # EXC002: catches everything, no raise
        cleanup()


def exc002_correct_specific() -> None:
    """CORRECT: Catch specific expected exceptions."""
    try:
        risky_operation()
    except ValueError as e:
        logger.exception(f'Invalid value: {e}')  # Proper logging with traceback
        return None  # OK - specific exception handled


def exc002_correct_reraise() -> None:
    """CORRECT: Cleanup then re-raise."""
    try:
        risky_operation()
    except Exception:
        cleanup()
        raise  # Re-raise to propagate


def exc002_correct_error_boundary() -> None:
    """CORRECT: Error boundary pattern (strong exception safety)."""
    try:
        risky_operation()
    except BaseException:
        rollback()
        raise  # Always re-raise after cleanup


def exc002_correct_with_note() -> None:
    """CORRECT: Error boundary with add_note() for context (Python 3.11+).

    This pattern adds context to the exception without wrapping it in a new
    exception type. The nested handler doesn't re-raise the cleanup error
    because we want the original exception to propagate, not the cleanup error.

    Note: The linter allows this pattern (nested handler in re-raising context).
    Using add_note() is best practice but not required - the linter won't enforce
    this level of strictness.
    """
    try:
        risky_operation()
    except BaseException as exc:
        # Perform cleanup
        try:
            rollback()
            exc.add_note('Rollback completed successfully')
        except Exception as rollback_error:
            # Not re-raised: we enrich the original exception with cleanup
            # failure info, not replace it with the cleanup error
            exc.add_note(f'Rollback failed: {rollback_error}')
        raise  # Original exception propagates with added notes


def exc002_correct_excepthook() -> None:
    """CORRECT: Entry-point error boundary using sys.excepthook.

    At program entry points there is no caller to propagate to. The traditional
    pattern of try/except Exception at __main__ triggers EXC002 because there's
    no re-raise. sys.excepthook is Python's canonical mechanism for this — the
    exception propagates naturally to the top level, Python calls your hook for
    formatting, and the process exits with code 1 (non-zero).

    This is NOT exception swallowing — the exception is:
    - Fully reported (traceback printed to stderr)
    - Process exits non-zero (OS sees the failure)
    - No try/except needed (no broad catch to flag)

    For thread entry points, use threading.excepthook (Python 3.8+).
    For unraisable exceptions (__del__, weak refs), use sys.unraisablehook.
    For rich formatting, libraries like rich.traceback.install() use this same
    mechanism: they replace sys.excepthook with a prettier formatter.

    Equivalent pattern in other languages:
    - Rust: std::panic::set_hook()
    - React: Error Boundary components
    """

    def _excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Custom exception formatter for entry-point errors."""
        # Let system exceptions (KeyboardInterrupt, SystemExit, GeneratorExit)
        # use default handling — issubclass(_, Exception) is the positive check
        # that covers all application errors without brittle enumeration
        if not issubclass(exc_type, Exception):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        # Format application errors
        print(f'Fatal: {exc_type.__name__}: {exc_value}', file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)

    sys.excepthook = _excepthook
    main()  # Exception propagates naturally — no try/except needed


def exc002_suppressed_intentional() -> None:
    """SUPPRESSED: Rare case where swallowing is intentional."""
    try:
        optional_feature()
    except Exception:  # exception_safety_linter.py: swallowed-exception
        # Feature is optional - failure is acceptable
        # Documented decision to continue without this feature
        pass


# =============================================================================
# EXC003: Control flow in finally (return/break/continue suppresses exceptions)
# =============================================================================


def exc003_violation_return() -> str:
    """VIOLATION: return in finally suppresses exceptions."""
    try:
        raise ValueError('Something went wrong')
    finally:
        return 'Success'  # EXC003: silently swallows ValueError!


def exc003_violation_break() -> None:
    """VIOLATION: break in finally suppresses exceptions."""
    items = ['a', 'b', 'c']  # Example items to process
    for item in items:
        try:
            process(item)
        finally:
            break  # EXC003: suppresses any exception from process()


def exc003_violation_continue() -> None:
    """VIOLATION: continue in finally suppresses exceptions."""
    items = ['a', 'b', 'c']  # Example items to process
    for item in items:
        try:
            process(item)
        finally:
            continue  # EXC003: suppresses any exception from process()


def exc003_correct_return() -> str | None:
    """CORRECT: Move return outside finally block."""
    result = None
    try:
        result = compute()
    finally:
        cleanup()
    return result  # Return AFTER finally


def exc003_correct_break() -> None:
    """CORRECT: Move break outside finally block."""
    items = ['a', 'b', 'c']  # Example items to process
    for item in items:
        should_break = False
        try:
            process(item)
            should_break = True
        finally:
            cleanup()
        if should_break:
            break


def exc003_correct_continue() -> None:
    """CORRECT: Move continue outside finally block."""
    items = ['a', 'b', 'c']  # Example items to process
    for item in items:
        should_continue = False
        try:
            if not validate(item):
                should_continue = True
            else:
                process(item)
        finally:
            cleanup()
        if should_continue:
            continue


def exc003_correct_contextmanager() -> str | None:
    """CORRECT: Use contextlib.contextmanager to avoid try/finally temptation.

    Context managers separate resource lifecycle from business logic. The
    decorator handles try/finally internally, so there's no opportunity for
    return/break/continue in a finally block.

    This is the structural fix — rather than moving control flow outside
    finally (which the other correct patterns show), eliminate the manual
    try/finally entirely.
    """

    @contextlib.contextmanager
    def managed_resource() -> Iterator[str]:
        resource = acquire_resource()
        try:
            yield resource
        finally:
            release_resource(resource)

    with managed_resource() as r:
        return compute_with(r)  # Clean — finally cleanup happens automatically


def exc003_suppressed_top_level() -> int:
    """SUPPRESSED: Rare top-level case where suppression is intended."""
    try:
        main()
    finally:
        return 0  # exception_safety_linter.py: finally-control-flow


# =============================================================================
# EXC004: Raise without from (implicit vs explicit exception chaining)
# =============================================================================
#
# IMPORTANT: Raising a new exception inside an except block does NOT lose
# the original traceback. Python automatically sets __context__ (implicit
# chaining). The difference between implicit and explicit chaining is:
#
#   Implicit (`raise NewError()`):
#     - Sets __context__ to original exception
#     - Traceback says: "During handling of the above exception, another
#       exception occurred"
#     - Suggests the new exception might be accidental/unexpected
#
#   Explicit (`raise NewError() from original`):
#     - Sets __cause__ to original exception
#     - Traceback says: "The above exception was the direct cause of the
#       following exception"
#     - Shows deliberate exception transformation
#
#   Suppressed (`raise NewError() from None`):
#     - Hides the original exception from traceback
#     - Use sparingly (e.g., hiding internal implementation details)
# =============================================================================


def exc004_violation_basic() -> None:
    """VIOLATION: Raising new exception without from uses implicit chaining.

    The original ValueError is preserved in __context__, but the traceback
    message "During handling of..." suggests this might be accidental.
    Use `from e` to show the transformation is intentional.
    """
    try:
        risky_operation()
    except ValueError:
        raise CustomError('Operation failed')  # EXC004: implicit chaining - unclear intent


def exc004_correct_explicit() -> None:
    """CORRECT: Use 'from e' to show deliberate exception transformation.

    Traceback will say: "The above exception was the direct cause of the
    following exception" - making it clear this is intentional wrapping.
    """
    try:
        risky_operation()
    except ValueError as e:
        raise CustomError('Operation failed') from e  # Explicit: __cause__ = e


def exc004_correct_suppress() -> None:
    """CORRECT: Use 'from None' to intentionally suppress the chain.

    Use when you want to hide internal implementation details from the
    public API surface. The original exception won't appear in the traceback.
    """
    try:
        internal_detail()
    except InternalError:
        raise PublicError('Operation failed') from None  # Intentionally suppressed


def exc004_correct_reraise() -> None:
    """CORRECT: Re-raise original exception (no chaining needed)."""
    try:
        risky_operation()
    except ValueError:
        cleanup()
        raise  # No new exception, just re-raise original


# NOTE: No suppression example for EXC004. The linter encourages explicit
# chaining (`from e`) or explicit suppression (`from None`). Both are valid
# depending on intent - the goal is to be explicit about which you mean.


# =============================================================================
# EXC005: Unused exception variable (captured but never used)
# =============================================================================


def exc005_violation_basic() -> None:
    """VIOLATION: Capturing 'as e' but never using e."""
    try:
        risky_operation()
    except ValueError as e:  # EXC005: e is captured but never used
        return None


def exc005_correct_no_capture() -> None:
    """CORRECT: Don't capture if not needed."""
    try:
        risky_operation()
    except ValueError:  # No 'as e' - clearer intent
        return None


def exc005_correct_use_variable() -> None:
    """CORRECT: Use the exception variable."""
    try:
        risky_operation()
    except ValueError as e:
        logger.exception(f'Failed: {e}')  # Using e with proper traceback (EXC006 compliant)
        return None


# =============================================================================
# EXC006: Logger without exc_info (loses traceback in logs)
# =============================================================================


def exc006_violation_basic() -> None:
    """VIOLATION: logger.error() without exc_info loses traceback."""
    try:
        risky_operation()
    except Exception as e:
        logger.error(f'Failed: {e}')  # EXC006: no exc_info - traceback lost!
        raise


def exc006_violation_critical() -> None:
    """VIOLATION: Works for all logger error methods."""
    try:
        risky_operation()
    except Exception:
        logger.critical('Critical failure')  # EXC006: no exc_info
        raise


def exc006_correct_exception() -> None:
    """CORRECT: Use logger.exception() (auto-includes exc_info)."""
    try:
        risky_operation()
    except Exception:
        logger.exception('Operation failed')  # Auto-includes traceback
        raise  # Proper propagation


def exc006_correct_exc_info() -> None:
    """CORRECT: Use exc_info=True explicitly."""
    try:
        risky_operation()
    except Exception as e:
        logger.error('Failed: %s', e, exc_info=True)  # Includes traceback
        raise


def exc006_correct_when_suppressing() -> None:
    """CORRECT: Even when suppressing (EXC002), preserve the traceback.

    Shows that logging rules apply regardless of exception handling strategy.
    If you must swallow an exception, at least log it with full context.
    The EXC002 suppression is documented - this is rare but justified for
    truly optional features where failure should not propagate.
    """
    try:
        optional_feature()
    except Exception:  # exception_safety_linter.py: swallowed-exception
        logger.exception('Optional feature failed, continuing')  # Traceback preserved


# =============================================================================
# EXC007: CancelledError not raised (breaks async cancellation)
# =============================================================================


async def exc007_violation_basic() -> None:
    """VIOLATION: Catching CancelledError without re-raising."""
    try:
        await async_operation()
    except asyncio.CancelledError:  # EXC007: no raise - breaks cancellation!
        cleanup()


async def exc007_violation_return_in_worker() -> None:
    """VIOLATION: CancelledError swallowed with return in worker loop.

    This is the most common form of this bug in queue workers.
    The developer thinks `return` cleanly exits the loop, but it actually:
    - Makes task.cancelled() return False (appears completed, not cancelled)
    - Breaks orchestrator logic that checks task.cancelled()
    - Silently corrupts the cancellation chain
    """
    queue: asyncio.Queue[object] = asyncio.Queue()
    while True:
        try:
            item = await queue.get()
        except asyncio.CancelledError:
            return  # EXC007: Swallows cancellation! Use `raise` or remove try/except
        await process_async(item)


async def exc007_correct_explicit() -> None:
    """CORRECT: Always re-raise CancelledError after cleanup."""
    try:
        await async_operation()
    except asyncio.CancelledError:
        logger.info('Task cancelled, cleaning up')
        cleanup()
        raise  # MUST re-raise to propagate cancellation


async def exc007_correct_three_way() -> ErrorResult | None:
    """CORRECT: Error boundary pattern (three-way classification).

    This is the gold standard for async exception handling:
    - CancelledError: cleanup + propagate
    - Expected errors: handle gracefully
    - Unexpected errors: cleanup + propagate
    """
    try:
        await async_operation()
    except asyncio.CancelledError:
        # Cancellation - cleanup and propagate
        cleanup()
        raise
    except ValueError as e:
        # Expected error - handle gracefully
        return ErrorResult(str(e))
    except Exception:
        # Unexpected error - cleanup and propagate
        cleanup()
        raise
    return None


async def exc007_correct_base_exception() -> None:
    """CORRECT: BaseException with re-raise is OK in async."""
    try:
        await async_operation()
    except BaseException:
        # Error boundary - handles CancelledError, KeyboardInterrupt, etc.
        await rollback_async()
        raise  # Always re-raise


async def exc007_correct_no_catch_needed() -> None:
    """CORRECT: Don't catch CancelledError when no cleanup is needed.

    The simplest and preferred pattern: let CancelledError propagate naturally.
    Only catch it when you have cleanup to perform before propagation.

    This is better than `except CancelledError: raise` which catches just to re-raise.
    """
    queue: asyncio.Queue[object] = asyncio.Queue()
    while True:
        item = await queue.get()  # CancelledError propagates automatically
        await process_async(item)


async def exc007_correct_taskgroup() -> None:
    """CORRECT: asyncio.TaskGroup handles cancellation automatically (Python 3.11+).

    TaskGroup is the structural alternative to manual three-way classification.
    When any task fails:
    - Sibling tasks are automatically cancelled
    - CancelledError propagates correctly without manual re-raise
    - Multiple failures are collected into an ExceptionGroup

    No try/except CancelledError needed — the TaskGroup protocol handles it.
    This is to EXC007 what contextlib.contextmanager is to EXC003: a structural
    fix that eliminates the anti-pattern rather than just correcting it.
    """
    async with asyncio.TaskGroup() as tg:
        tg.create_task(task_a())
        tg.create_task(task_b())
        # If either task raises, the other is cancelled automatically.
        # CancelledError propagation is handled by TaskGroup.


# =============================================================================
# EXC008: GeneratorExit not raised (breaks generator cleanup protocol)
# =============================================================================


def exc008_violation_basic() -> Generator[int]:
    """VIOLATION: Catching GeneratorExit without re-raising.

    When generator.close() is called, GeneratorExit is raised. If caught
    without re-raising, close() cannot complete properly, causing:
    - Context managers inside generator don't exit cleanly
    - Resources leak (file handles, connections, locks)
    - Generator cleanup protocol broken
    """
    value = 42
    try:
        yield value
    except GeneratorExit:  # EXC008: no raise - generator.close() incomplete!
        cleanup()


def exc008_correct_explicit() -> Generator[int]:
    """CORRECT: Always re-raise GeneratorExit after cleanup.

    Like CancelledError in async (EXC007), GeneratorExit must propagate
    to maintain the cleanup protocol. Catch only when you have cleanup
    to perform, then re-raise.
    """
    value = 42
    try:
        yield value
    except GeneratorExit:
        logger.info('Generator closing, cleaning up')
        cleanup()
        raise  # MUST re-raise to complete close()


def exc008_correct_finally() -> Generator[int]:
    """CORRECT: Use finally instead (preferred pattern).

    Unlike try/except GeneratorExit, finally always runs regardless of
    how the generator exits (normal completion, close(), throw(), etc.).
    This is the cleanest pattern when you just need cleanup.

    Mirrors exc007_correct_no_catch_needed for async functions.
    """
    value = 42
    try:
        yield value
    finally:
        cleanup()  # Runs automatically on close()


def exc008_correct_contextmanager() -> None:
    """CORRECT: contextlib.contextmanager handles GeneratorExit implicitly.

    The @contextmanager decorator wraps a generator in a context manager that
    handles the cleanup protocol (including GeneratorExit) internally. You
    write try/yield/finally and the decorator does the rest.

    This is to EXC008 what TaskGroup is to EXC007: a structural fix that
    eliminates the manual protocol handling entirely. The generator is never
    directly exposed — callers use 'with' instead of next()/close().
    """

    @contextlib.contextmanager
    def managed_connection() -> Iterator[object]:
        conn = acquire_resource()
        try:
            yield conn
        finally:
            release_resource(conn)  # GeneratorExit handled by decorator

    with managed_connection() as conn:
        process(conn)


# =============================================================================
# Combined Example: Error Boundary Pattern (Strong Exception Safety)
# =============================================================================


async def error_boundary_example() -> ErrorResult | SuccessResult:
    """Demonstrates the error boundary pattern with atomic rollback.

    This is the gold standard pattern that the linter enforces.
    Three-way exception classification:
    - Cancellation: cleanup + re-raise
    - Expected errors: handle gracefully
    - Unexpected errors: cleanup + re-raise

    Uses add_note() (Python 3.11+) to enrich exceptions with context.
    """
    backup_path = Path('/tmp/backup')
    await create_backup(backup_path)

    try:
        # Attempt risky operation
        await modify_data()

    except asyncio.CancelledError as exc:
        # Cancellation - cleanup and propagate
        logger.info('Operation cancelled, rolling back')
        await rollback_async(backup_path)
        exc.add_note('Rollback completed after cancellation')
        raise  # MUST re-raise

    except ValueError as e:
        # Expected error - handle gracefully
        logger.error('Invalid data: %s', e, exc_info=True)
        await rollback_async(backup_path)
        return ErrorResult(str(e))

    except BaseException as exc:
        # Unexpected error - cleanup and propagate
        logger.error('Unexpected error, rolling back', exc_info=True)
        try:
            await rollback_async(backup_path)
            exc.add_note('Rollback completed successfully')
        except Exception as rollback_error:
            # Not re-raised: we enrich the original exception with cleanup
            # failure info, not replace it with the cleanup error
            exc.add_note(f'Rollback failed: {rollback_error}')
            exc.add_note(f'Backup preserved at: {backup_path}')
        raise  # Let unexpected errors propagate with full context

    # Success - clean up backup
    backup_path.unlink()
    return SuccessResult()


# =============================================================================
# Private Helper Functions (for examples above)
# =============================================================================


# -----------------------------------------------------------------------------
# Exception Classes
# -----------------------------------------------------------------------------


class CustomError(Exception):
    """Custom exception for examples."""

    pass


class InternalError(Exception):
    """Internal exception for examples."""

    pass


class PublicError(Exception):
    """Public exception for examples."""

    pass


# -----------------------------------------------------------------------------
# Result Classes
# -----------------------------------------------------------------------------


class ErrorResult:
    """Placeholder for error result."""

    def __init__(self, message: str) -> None:
        self.message = message


class SuccessResult:
    """Placeholder for success result."""

    pass


# -----------------------------------------------------------------------------
# Stub Functions (alphabetical)
# -----------------------------------------------------------------------------


def acquire_resource() -> str:
    """Placeholder for resource acquisition."""
    return ''


async def async_operation() -> None:
    """Placeholder for async operation."""
    pass


def cleanup() -> None:
    """Placeholder for cleanup logic."""
    pass


def compute() -> str:
    """Placeholder for computation."""
    return ''


def compute_with(resource: object) -> str:
    """Placeholder for computation with resource."""
    return ''


async def create_backup(path: Path) -> None:
    """Placeholder for backup creation."""
    pass


def handle_error() -> None:
    """Placeholder for error handling."""
    pass


def internal_detail() -> None:
    """Placeholder for internal operation."""
    pass


def main() -> None:
    """Placeholder for main entry point."""
    pass


async def modify_data() -> None:
    """Placeholder for data modification."""
    pass


def optional_feature() -> None:
    """Placeholder for optional feature."""
    pass


def process(item: object) -> None:
    """Placeholder for sync item processing."""
    pass


async def process_async(item: object) -> None:
    """Placeholder for async item processing."""
    pass


def release_resource(resource: object) -> None:
    """Placeholder for resource release."""
    pass


def risky_operation() -> None:
    """Placeholder for operation that might raise."""
    pass


def rollback() -> None:
    """Placeholder for rollback logic."""
    pass


async def rollback_async(backup_path: Path | None = None) -> None:
    """Placeholder for async rollback logic."""
    pass


async def task_a() -> None:
    """Placeholder for async task."""
    pass


async def task_b() -> None:
    """Placeholder for async task."""
    pass


def validate(item: str) -> bool:
    """Placeholder for validation."""
    return True
