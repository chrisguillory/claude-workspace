"""Module-level Exception subclasses for the EXC010 empirical test.

The test framework runs the linter on this file; the EXPECTED dict in
``test_exception_safety_linter.py`` asserts which classes fire EXC010.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pydantic
from cc_lib.picklable import PickleByInitArgs

# -- Should fire EXC010 ------------------------------------------------------


class KwOnlyError_pickle_breaks(Exception):
    """kw-only init args can't survive pickle's positional reconstruction."""

    def __init__(self, *, name: str) -> None:
        super().__init__(name)
        self.name = name


class KwargsError_pickle_breaks(Exception):
    """**kwargs init can't survive pickle round-trip."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__('error')
        self.kwargs = kwargs


class TransformedMessageError_pickle_breaks(Exception):
    """super() receives a derived value; pickle reconstructs from the formatted message."""

    def __init__(self, var_name: str) -> None:
        super().__init__(f'{var_name} is broken')
        self.var_name = var_name


class ArgCountMismatchError_pickle_breaks(Exception):
    """__init__ drops a param before calling super()."""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


# -- Should NOT fire EXC010 --------------------------------------------------


class PassthroughError_pickle_ok(Exception):
    """__init__ passes its only param verbatim to super()."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


class MultiPassthroughError_pickle_ok(Exception):
    """__init__ passes all params verbatim, in order, to super()."""

    def __init__(self, code: int, detail: str) -> None:
        super().__init__(code, detail)
        self.code = code
        self.detail = detail


class ReduceEscapeHatchError_pickle_ok(Exception):
    """Defines __reduce__ as the explicit pickle escape hatch."""

    def __init__(self, var_name: str) -> None:
        super().__init__(f'{var_name} broken')
        self.var_name = var_name

    def __reduce__(self) -> tuple[type[ReduceEscapeHatchError_pickle_ok], tuple[str]]:
        return (self.__class__, (self.var_name,))


class NoInitError_pickle_ok(Exception):
    """No custom __init__; inherits Exception's default."""


class VarargPassthroughError_pickle_ok(Exception):
    """super() receives the vararg as a starred unpack."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class NoSuperCallError_pickle_ok(Exception):
    """No super().__init__() call but pickle still round-trips.

    Exception.__new__ stores positional args in self.args before __init__ runs,
    so unpickle replays the constructor and recovers the same state.
    """

    def __init__(self, value: int) -> None:
        self.value = value


# -- Breaks pickle but silenced by an inline directive — should NOT fire EXC010 --


class InlineSuppressedError_pickle_breaks(Exception):  # exception_safety_linter.py: init-not-pickleable
    """Same shape as TransformedMessageError, but the inline directive suppresses EXC010.

    Contrast with the unsuppressed twin proves the empirical checker honors
    inline directives (not just config per-file-ignores).
    """

    def __init__(self, var_name: str) -> None:
        super().__init__(f'{var_name} is broken')
        self.var_name = var_name


# -- Mixin + synthesis coverage — should NOT fire EXC010 ---------------------


class MixinSeqLiteralError_pickle_ok(PickleByInitArgs, Exception):
    """Keyword-only init with an iterated Sequence[Path] and a Literal.

    The mixin round-trips it; synthesis must yield [] for the sequence (else the
    iteration raises) and a Literal member (else calling the string value raises).
    """

    def __init__(self, *, name: str, paths: Sequence[Path], mode: Literal['fast', 'slow']) -> None:
        self.name = name
        self.paths = paths
        self.mode = mode
        joined = ', '.join(str(p) for p in paths)
        super().__init__(f'{name} [{mode}]: {joined}')


class WrappedCauseError_pickle_ok(PickleByInitArgs, Exception):
    """Stores an Exception attribute (no value-equality).

    The round-trip yields an equivalent-but-distinct cause, so attribute state must
    be compared by pickle bytes, not ``==``, to avoid a false positive.
    """

    def __init__(self, cause: Exception) -> None:
        self.cause = cause
        super().__init__(f'wrapped: {cause}')


class ValidationErrorParamError_pickle_ok(PickleByInitArgs, Exception):
    """Takes a pydantic.ValidationError, which has no no-arg form — synthesis mints one via its factory."""

    def __init__(self, error: pydantic.ValidationError) -> None:
        self.error = error
        super().__init__(str(error))
