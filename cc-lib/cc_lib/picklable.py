"""Pickle support for exceptions whose custom ``__init__`` doesn't match ``self.args``."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

__all__ = [
    'PickleByInitArgs',
]


class PickleByInitArgs:
    """Mixin making an exception with a custom ``__init__`` picklable.

    ``BaseException.__reduce__`` reconstructs via ``cls(*self.args)`` — the
    formatted message — which a custom (especially keyword-only) ``__init__``
    cannot accept, so the round-trip raises or double-formats. This mixin instead
    reconstructs from the ``__init__`` parameters, read back from same-named
    attributes, preserving message, attrs, and post-construction ``__dict__`` state.

    List it FIRST in the bases (``class E(PickleByInitArgs, Base)``) so its
    ``__reduce__`` precedes ``BaseException``'s in the MRO. Requirement: each
    ``__init__`` parameter is stored *unmodified* under its own name. A missing
    attribute raises ``AttributeError`` at pickle time (loud); a derived value
    (``self.x = f(x)``) silently corrupts the round-trip, so store the raw
    parameter and derive elsewhere.
    """

    def __reduce__(self) -> tuple[Any, ...]:
        params = inspect.signature(type(self).__init__).parameters
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for name, param in params.items():
            if name == 'self' or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            value = getattr(self, name)
            if param.kind is param.KEYWORD_ONLY:
                kwargs[name] = value
            else:
                args.append(value)
        return (_reconstruct, (type(self), tuple(args), kwargs), self.__dict__)


def _reconstruct(cls: type[Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    """Rebuild a PickleByInitArgs exception during unpickling (pickle delivers args positionally)."""
    return cls(*args, **kwargs)
