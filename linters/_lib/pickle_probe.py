"""Pickle probe for EXC010.

Imports a target file in THIS interpreter's environment, pickle-round-trips its
Exception subclasses, and emits per-class diagnostics as JSON on stdout.

Run as a subprocess so the target's module-level code (which may ``sys.exit``,
read stdin, or import deps absent from the linter's env) cannot corrupt or kill
the linter process. The interpreter is chosen by the parent: the workspace
``sys.executable`` for ordinary files, or ``uv python find --script``'s cached
environment for PEP 723 scripts whose deps live outside the workspace.

No top-level third-party imports, so the probe starts under whatever interpreter
the parent hands it (including a PEP 723 script's lean cached env). Value
synthesis may still use the target's own types — already in hand via each
``__init__`` signature — to mint awkward arguments (e.g. a ``pydantic``
``ValidationError`` via its factory) without importing anything here.

Usage: python pickle_probe.py <target_file> [<sys_path_root> ...]
Output: a ``ProbeOutput`` as JSON — one ``ProbeResult`` per module-level Exception
subclass. The parent imports those entities and validates the payload, so their
fields (not a hand-drawn shape here) are the single source of truth for the wire.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import importlib
import inspect
import json
import pickle
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints

__all__ = [
    'ProbeOutput',
    'ProbeResult',
    'diagnose_file',
]


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """One Exception subclass's pickle-round-trip outcome.

    ``diagnostic`` is the round-trip failure (the violation), or ``None`` when the
    class round-trips cleanly. ``unsynthesizable`` is set instead when no instance
    could be constructed to test — disclosed, never silently dropped.

    The parent validates this via ``pydantic.TypeAdapter``; the ``forbid``/``strict``
    policy below mirrors ``cc_lib``'s ``ClosedModel`` (internal data — reject unknown
    fields) but rides as a plain dict so the stdlib-only probe needs no pydantic import.
    """

    __pydantic_config__: ClassVar[Mapping[str, object]] = {'extra': 'forbid', 'strict': True}

    class_name: str
    diagnostic: str | None = None
    unsynthesizable: str | None = None


@dataclasses.dataclass(frozen=True)
class ProbeOutput:
    """The probe's per-file result — one ``ProbeResult`` per Exception subclass."""

    __pydantic_config__: ClassVar[Mapping[str, object]] = {'extra': 'forbid', 'strict': True}

    classes: Sequence[ProbeResult]


def diagnose_file(filepath: Path, roots: Sequence[str]) -> Sequence[ProbeResult]:
    """Import ``filepath`` and pickle-round-trip each Exception subclass it defines.

    A class whose ``__init__`` needs an argument that cannot be synthesized from
    its type hint is disclosed as ``unsynthesizable`` (and skipped), so one
    un-constructible signature does not blind the rule to its siblings.
    """
    sys.path[:0] = list(roots)
    module = _import_module(filepath)
    results: list[ProbeResult] = []
    for cls_name, cls in inspect.getmembers(module, inspect.isclass):
        if not issubclass(cls, BaseException):
            continue
        if cls.__module__ != module.__name__:
            continue
        try:
            instance = _construct(cls)
        except TypeError as exc:
            results.append(ProbeResult(class_name=cls_name, unsynthesizable=f'{type(exc).__name__}: {exc}'))
            continue
        results.append(ProbeResult(class_name=cls_name, diagnostic=_diagnose(instance)))
    return results


def main() -> int:
    """Probe the target file named in argv and write diagnostics as JSON."""
    target = Path(sys.argv[1])
    roots = sys.argv[2:]
    results = diagnose_file(target, roots)
    json.dump(dataclasses.asdict(ProbeOutput(classes=list(results))), sys.stdout)
    return 0


def _import_module(filepath: Path) -> ModuleType:
    """Import ``filepath`` by qualified name so its package initializes in order.

    ``import_module`` (vs ``spec_from_file_location``) runs parent ``__init__``
    first, so re-export chains (``from .base import X`` in a package ``__init__``)
    resolve instead of hitting a partially-initialized submodule.
    """
    root = filepath.resolve().parent
    while (root / '__init__.py').exists() and root.parent != root:
        root = root.parent
    module_name = '.'.join(filepath.resolve().relative_to(root).with_suffix('').parts)
    return importlib.import_module(module_name)


def _diagnose(instance: BaseException) -> str | None:
    """Pickle-round-trip ``instance``. Returns a diagnostic if the round-trip is lossy."""
    try:
        restored = pickle.loads(pickle.dumps(instance))
    except (pickle.PickleError, TypeError, AttributeError) as exc:
        return f'pickle round-trip raises {type(exc).__name__}: {exc}'
    if str(instance) != str(restored):
        return f'message changed across pickle: {str(instance)!r} -> {str(restored)!r}'
    if type(instance) is not type(restored):
        return 'type identity changed across pickle'
    # Compare attribute state by pickle-equivalence, not ``==``/identity: an attr
    # holding an object without ``__eq__`` (e.g. a wrapped ``cause`` exception)
    # is value-preserved across the round-trip but is a fresh instance, so ``==``
    # would false-positive. Equal pickle bytes ⇒ equivalent reconstructed state.
    if pickle.dumps(dict(vars(instance))) != pickle.dumps(dict(vars(restored))):
        return f'attribute state changed: {dict(vars(instance))} -> {dict(vars(restored))}'
    return None


def _construct(cls: type[BaseException]) -> BaseException:
    """Instantiate ``cls`` using type-hint-synthesized values for each parameter."""
    sig = inspect.signature(cls.__init__)
    hints = get_type_hints(cls.__init__)
    positional: list[Any] = []
    keyword: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name not in hints:
            msg = f'{cls.__module__}.{cls.__qualname__}.__init__ param {name!r} has no type hint'
            raise TypeError(msg)
        value = _synthesize_value(hints[name])
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            keyword[name] = value
        else:
            positional.append(value)
    instance = cls(*positional, **keyword)
    if not isinstance(instance, BaseException):
        msg = f'{cls.__module__}.{cls.__qualname__} did not produce a BaseException instance'
        raise TypeError(msg)
    return instance


def _synthesize_value(type_hint: Any) -> Any:
    """Synthesize a value of ``type_hint`` to construct (then pickle) the exception.

    Reliable for no-arg-constructible shapes: ``Literal`` (a member), unions
    (first non-None arm), parametrized containers (``Sequence[T]`` → ``[]``,
    ``Mapping`` → ``{}``), and plain types (``str`` → ``''``, ``Path`` → ``Path()``).

    A type that demands real arguments (e.g. ``pydantic.ValidationError``) raises
    ``TypeError`` from its no-arg constructor; that propagates so the caller records
    the class as un-analyzable rather than fabricate a value that could misrepresent
    the constructor — the round-trip is only as honest as the instance pickled.
    """
    if type_hint is type(None):
        return None
    origin = get_origin(type_hint)
    if origin is Literal:
        args = get_args(type_hint)
        return args[0] if args else None
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(type_hint) if a is not type(None)]
        return _synthesize_value(non_none[0]) if non_none else None
    if origin is not None:
        return _empty_for_origin(origin)
    try:
        return type_hint()
    except TypeError:
        return _synthesize_via_factory(type_hint)


def _synthesize_via_factory(type_hint: Any) -> Any:
    """Mint a representative instance of a type that has no no-arg constructor.

    Uses the type's own factory — the type is already imported via the target's
    ``__init__`` signature, so nothing is imported here. Extend with new cases as
    un-constructible parameter types appear in real exceptions; a type with no
    known factory raises ``TypeError`` so the caller discloses it as un-analyzable.
    """
    # pydantic ValidationError: build an empty one via its documented factory.
    from_exception_data = getattr(type_hint, 'from_exception_data', None)
    if callable(from_exception_data):
        return from_exception_data('probe', [])
    msg = f'{type_hint!r} has no no-arg constructor and no known synthesis factory'
    raise TypeError(msg)


def _empty_for_origin(origin: Any) -> Any:
    """An empty instance of a parametrized container's origin (``list[T]`` → ``[]``)."""
    if origin in (list, collections.abc.Sequence, collections.abc.MutableSequence, collections.abc.Iterable):
        return []
    if origin is tuple:
        return ()
    if origin in (set, frozenset, collections.abc.Set, collections.abc.MutableSet):
        return set()
    if origin in (dict, collections.abc.Mapping, collections.abc.MutableMapping):
        return {}
    return origin()


if __name__ == '__main__':
    sys.exit(main())
