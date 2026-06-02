# exception_safety_linter.py is the linter under test; the others skip this file.
# strict_typing_linter.py: skip-file
# suppression_rationale_linter.py: skip-file
# ruff: noqa: E722, F841, SIM105, TRY203
"""``used_``/``unused_`` fixtures for exception_safety_linter --report-unused-directives.

Each entity carries exactly one suppression directive. A ``used_*`` entity triggers its
rule, so the directive suppresses a real violation and must NOT be flagged unused; an
``unused_*`` entity does not trigger it, so the directive matches nothing and must be
flagged. Polarity is declared by the entity name; the EXPECTED map in
``test_unused_directives.py`` binds each entity to the code it exercises.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator

logger = logging.getLogger(__name__)


# -- bare-except (EXC001) -----------------------------------------------------


def used_bare_except() -> None:
    try:
        pass
    except:  # exception_safety_linter.py: bare-except
        pass


def unused_bare_except() -> None:
    try:
        pass
    except ValueError:  # exception_safety_linter.py: bare-except
        pass


# -- swallowed-exception (EXC002) ---------------------------------------------


def used_swallowed_exception() -> None:
    try:
        pass
    except Exception:  # exception_safety_linter.py: swallowed-exception
        pass


def unused_swallowed_exception() -> None:
    try:
        pass
    except Exception:  # exception_safety_linter.py: swallowed-exception
        raise


# -- finally-control-flow (EXC003) --------------------------------------------


def used_finally_control_flow() -> str:
    try:
        pass
    finally:
        return 'done'  # exception_safety_linter.py: finally-control-flow


def unused_finally_control_flow() -> None:
    try:
        pass
    finally:
        pass  # exception_safety_linter.py: finally-control-flow


# -- raise-without-from (EXC004) ----------------------------------------------


def used_raise_without_from() -> None:
    try:
        pass
    except ValueError:
        raise RuntimeError('boom')  # exception_safety_linter.py: raise-without-from


def unused_raise_without_from() -> None:
    try:
        pass
    except ValueError as e:
        raise RuntimeError('boom') from e  # exception_safety_linter.py: raise-without-from


# -- unused-exception-var (EXC005) --------------------------------------------


def used_unused_exception_var() -> None:
    try:
        pass
    except ValueError as e:  # exception_safety_linter.py: unused-exception-var
        return


def unused_unused_exception_var() -> None:
    try:
        pass
    except ValueError:  # exception_safety_linter.py: unused-exception-var
        return


# -- logger-no-exc-info (EXC006) ----------------------------------------------


def used_logger_no_exc_info() -> None:
    try:
        pass
    except Exception:
        logger.error('failed')  # exception_safety_linter.py: logger-no-exc-info
        raise


def unused_logger_no_exc_info() -> None:
    try:
        pass
    except Exception:
        logger.exception('failed')  # exception_safety_linter.py: logger-no-exc-info
        raise


# -- cancelled-not-raised (EXC007) --------------------------------------------


async def used_cancelled_not_raised() -> None:
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:  # exception_safety_linter.py: cancelled-not-raised
        pass


async def unused_cancelled_not_raised() -> None:
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError:  # exception_safety_linter.py: cancelled-not-raised
        raise


# -- generator-exit-not-raised (EXC008) ---------------------------------------


def used_generator_exit_not_raised() -> Generator[int]:
    try:
        yield 1
    except GeneratorExit:  # exception_safety_linter.py: generator-exit-not-raised
        pass


def unused_generator_exit_not_raised() -> Generator[int]:
    try:
        yield 1
    except GeneratorExit:  # exception_safety_linter.py: generator-exit-not-raised
        raise


# -- init-not-pickleable (EXC010) ---------------------------------------------


class UsedInitNotPickleable(Exception):  # exception_safety_linter.py: init-not-pickleable
    """Breaks pickle (super() gets a derived message); the directive suppresses EXC010."""

    def __init__(self, var_name: str) -> None:
        super().__init__(f'{var_name} is broken')
        self.var_name = var_name


class UnusedInitNotPickleable(Exception):  # exception_safety_linter.py: init-not-pickleable
    """Pickles cleanly (verbatim passthrough); the directive matches nothing."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg
