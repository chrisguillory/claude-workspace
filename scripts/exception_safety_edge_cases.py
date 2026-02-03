# ruff: noqa: SIM105
# SIM105: try-except-pass patterns required for testing; except* incompatible with contextlib.suppress
"""Exception safety edge cases - regression testing and false positive prevention.

This file complements exception_safety_test_cases.py with:
- Edge cases for unusual but valid Python patterns
- False positive prevention (valid code that should NOT trigger rules)
- Comprehensive coverage of all linter code paths

Unlike the instructive test file, functions here may test patterns that are
less common or demonstrate behaviors that don't fit the "one rule per function"
pedagogical model.

Run: ./validate_exception_linter.py (validates both test files)
"""

from __future__ import annotations

import logging
from asyncio import CancelledError
from asyncio import CancelledError as CE

logger = logging.getLogger(__name__)


# =============================================================================
# EXC002: Tuple Exception Edge Cases
# =============================================================================


def edge_tuple_with_broad_exception() -> None:
    """VIOLATION: Tuple containing Exception triggers EXC002.

    The linter checks each element of the tuple for broad exceptions.
    Even if specific exceptions are also listed, Exception/BaseException
    in the tuple makes it a broad catch.
    """
    try:
        risky()
    except (ValueError, Exception):  # EXC002: Exception in tuple
        pass


# =============================================================================
# EXC002: Non-Canonical Anti-Patterns (without pass)
# =============================================================================
# These verify the linter catches swallowed exceptions regardless of body content


def edge_swallowed_with_return() -> None:
    """VIOLATION: Non-canonical swallowed exception with return.

    The linter should catch this even though it's not try-except-pass.
    """
    try:
        risky()
    except Exception:  # EXC002: no raise, returns instead
        return


def edge_swallowed_with_action() -> None:
    """VIOLATION: Non-canonical swallowed exception with action.

    Does cleanup but doesn't re-raise - still a swallowed exception.
    """
    try:
        risky()
    except Exception:  # EXC002: cleanup without raise
        cleanup()


# =============================================================================
# EXC003: Finally Block Edge Cases
# =============================================================================


def edge_finally_with_raise() -> None:
    """CORRECT: raise in finally is not EXC003.

    EXC003 only flags return/break/continue (silent suppression).
    raise is explicit exception replacement, not hidden behavior.
    """
    try:
        risky()
    finally:
        raise RuntimeError('Cleanup failed')  # NOT EXC003


# =============================================================================
# EXC004: False Positive Prevention
# =============================================================================


def edge_raise_caught_variable() -> None:
    """CORRECT: Raising caught variable is NOT a new exception.

    When we write `raise e`, we're re-raising the same exception object.
    This is equivalent to bare `raise` - no new exception is created.
    The linter should NOT flag this as EXC004.
    """
    try:
        risky()
    except ValueError as e:
        cleanup()
        raise e  # NOT EXC004: re-raising caught exception


def edge_raise_caught_with_explicit_chain() -> None:
    """CORRECT: Re-raising with explicit 'from' is always valid.

    This tests that the linter doesn't flag `raise e from ...` patterns
    when e is the caught variable.
    """
    try:
        risky()
    except ValueError as e:
        raise e from None  # NOT EXC004: explicit chaining


def edge_raise_outer_scope_variable() -> None:
    """VIOLATION: Raising exception from outer scope IS EXC004.

    Triggers BOTH rules:
    - EXC004: The `raise outer` has unclear chaining intent
    - EXC002: The outer handler has no direct re-raise (nested raise not visible)

    The traceback will confusingly show:
        "During handling of [inner exception], another exception occurred"

    The fix is `raise outer from None` to explicitly suppress the chain,
    or restructure to avoid this pattern.
    """
    try:
        risky()
    except Exception as outer:
        try:
            cleanup()
        except ValueError:
            raise outer  # EXC004: unclear intent, use 'from' to clarify


# =============================================================================
# EXC006: All Logger Method Coverage
# =============================================================================


def edge_logger_fatal() -> None:
    """VIOLATION: logger.fatal() without exc_info loses traceback."""
    try:
        risky()
    except Exception:
        logger.fatal('Critical failure')  # EXC006
        raise


def edge_logger_warning() -> None:
    """VIOLATION: logger.warning() without exc_info loses traceback."""
    try:
        risky()
    except Exception:
        logger.warning('Something went wrong')  # EXC006
        raise


def edge_logger_warn() -> None:
    """VIOLATION: logger.warn() without exc_info loses traceback.

    Note: logger.warn() is deprecated in favor of logger.warning(),
    but the linter still checks it.
    """
    try:
        risky()
    except Exception:
        logger.warn('Deprecated warning method')  # EXC006
        raise


def edge_logger_exc_info_false() -> None:
    """VIOLATION: exc_info=False still triggers EXC006.

    Explicitly passing False still loses the traceback.
    Use suppression directive if this is intentional.
    """
    try:
        risky()
    except Exception:
        logger.error('Failed', exc_info=False)  # EXC006: False is not True
        raise


def edge_logger_info_no_violation() -> None:
    """CORRECT: logger.info() is not flagged (info is not error/warning level).

    EXC006 only applies to error-level methods (error, critical, fatal)
    and warning-level methods (warning, warn). Info and debug are not flagged.
    """
    try:
        risky()
    except Exception:
        logger.info('Informational message during exception handling')
        raise


def edge_logger_suppressed() -> None:
    """SUPPRESSED: Intentionally omitting traceback in specific log call.

    Use when the full traceback would be redundant (e.g., logged elsewhere)
    or when you want a cleaner log message for expected errors.
    """
    try:
        risky()
    except ValueError:
        logger.error('Expected error occurred')  # exception_safety_linter.py: logger-no-exc-info
        raise


# =============================================================================
# EXC007: BaseException in Async
# =============================================================================


async def edge_async_base_exception_no_raise() -> None:
    """VIOLATION: BaseException in async catches CancelledError.

    Triggers BOTH rules (they're related):
    - EXC007: Catches CancelledError without re-raise (async-specific)
    - EXC002: Broad exception (BaseException) without re-raise (general)

    Since Python 3.8, CancelledError inherits from BaseException.
    Catching BaseException in async code without re-raise breaks cancellation.
    """
    try:
        await async_op()
    except BaseException:  # EXC007: catches CancelledError, no raise
        cleanup()


async def edge_async_base_exception_with_raise() -> None:
    """CORRECT: BaseException with re-raise propagates cancellation."""
    try:
        await async_op()
    except BaseException:
        cleanup()
        raise  # Proper propagation


async def edge_cancelled_error_short_name() -> None:
    """VIOLATION: Short name CancelledError also triggers EXC007.

    Whether using asyncio.CancelledError or importing CancelledError
    directly, catching without re-raise breaks cancellation.
    """
    try:
        await async_op()
    except CancelledError:  # EXC007: short name, no raise
        cleanup()


async def edge_cancelled_error_aliased() -> None:
    """VIOLATION: Aliased CancelledError also triggers EXC007.

    The linter tracks import aliases and correctly identifies
    that CE refers to asyncio.CancelledError.
    """
    try:
        await async_op()
    except CE:  # EXC007: aliased import, no raise
        cleanup()


async def edge_cancelled_error_in_tuple() -> None:
    """VIOLATION: CancelledError in tuple without re-raise.

    When catching CancelledError as part of a tuple, the linter correctly
    identifies that cancellation semantics could be broken.
    """
    try:
        await async_op()
    except (ValueError, CancelledError):  # EXC007: CancelledError in tuple, no raise
        cleanup()


async def edge_cancelled_error_with_return() -> None:
    """VIOLATION: return instead of raise in CancelledError handler.

    This looks like a clean exit but breaks the cancellation chain:
    - task.cancelled() returns False (appears completed, not cancelled)
    - Orchestrator code checking task.cancelled() will be confused
    - The task appears to have completed normally, not been cancelled

    Common in worker loops: `while True: try: await queue.get() except CancelledError: return`
    Fix: Use `raise` after cleanup, or remove the try/except entirely if no cleanup needed.
    """
    try:
        await async_op()
    except CancelledError:
        return  # EXC007: use `raise` or remove try/except entirely


async def edge_sync_nested_in_async() -> None:
    """CORRECT: Sync function inside async doesn't trigger EXC007.

    A synchronous function defined inside an async function is not subject
    to async cancellation rules. Catching CancelledError in sync context
    is unusual but doesn't break cancellation semantics.
    """

    def sync_handler() -> None:
        try:
            risky()
        except CancelledError:  # NOT EXC007: sync context
            cleanup()  # Unusual but not a violation

    sync_handler()


# =============================================================================
# TryStar (Python 3.11+ Exception Groups)
# =============================================================================


def edge_trystar_no_raise() -> None:
    """VIOLATION: except* with broad exception and no raise.

    Exception groups (PEP 654) use except* syntax. The same rules apply:
    broad catches without re-raise are flagged.
    """
    try:
        risky()
    except* Exception:  # EXC002: broad catch without raise
        pass


def edge_trystar_with_raise() -> None:
    """CORRECT: except* with re-raise is proper error handling."""
    try:
        risky()
    except* Exception:
        cleanup()
        raise  # Proper re-raise in exception group


def edge_trystar_specific() -> None:
    """CORRECT: except* with specific exception is fine."""
    try:
        risky()
    except* ValueError:
        pass  # OK: specific exception


# =============================================================================
# Suppression Directive Edge Cases
# =============================================================================


async def edge_multi_code_suppression() -> None:
    """SUPPRESSED: Multiple codes on one directive line.

    Both swallowed-exception (EXC002) and cancelled-not-raised (EXC007)
    are suppressed with a single comma-separated directive.
    """
    try:
        await async_op()
    except BaseException:  # exception_safety_linter.py: swallowed-exception, cancelled-not-raised
        pass


# =============================================================================
# Stub Functions
# =============================================================================


async def async_op() -> None:
    """Placeholder for async operation."""
    pass


def cleanup() -> None:
    """Placeholder for cleanup logic."""
    pass


def risky() -> None:
    """Placeholder for operation that might raise."""
    pass
