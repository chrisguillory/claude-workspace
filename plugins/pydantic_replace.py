"""Mypy plugin that adds typed ``__replace__()`` to Pydantic models.

The problem
-----------
Pydantic frozen models need a way to create modified copies. The standard
approach is ``model_copy(update={'field': value})``, but ``update`` is typed as
``dict[str, Any]`` — mypy cannot catch typo'd field names or wrong value types.

Python 3.13 introduced ``__replace__()`` (PEP 681) which type checkers can
synthesize with per-field signatures. Pydantic v2.10+ implements it at runtime.
However, the pydantic mypy plugin strips ``@dataclass_transform()`` from
``ModelMetaclass`` to handle field aliases correctly, and this is exactly what
mypy needs to synthesize ``__replace__()``. The result: mypy sees no
``__replace__`` on Pydantic models, and the only type-safe copy mechanism is
unavailable.

The ``debug_dataclass_transform = true`` config restores the transform spec,
but mypy's built-in handler then takes over ``__init__`` synthesis too — and it
doesn't understand ``populate_by_name``, breaking models that use field aliases.

What this plugin does
---------------------
Inherits all pydantic plugin behavior (aliases, frozen detection,
``init_forbid_extra``) and adds ``__replace__()`` synthesis after each model
class is processed. No code is copied — the plugin subclasses ``PydanticPlugin``
and chains one additional step onto ``get_base_class_hook``.

What this catches
-----------------
.. code-block:: python

    result.__replace__(timming=None)   # error: No parameter named "timming"
    result.__replace__(timing="oops")  # error: "str" not assignable to "float | None"

Without this plugin, ``model_copy(update={'timming': None})`` silently accepts
both errors because ``update`` is ``dict[str, Any]``.

Configuration
-------------
Replace ``pydantic.mypy`` in your mypy plugins list::

    [tool.mypy]
    plugins = ["plugins.pydantic_replace"]

    [tool.pydantic-mypy]
    init_forbid_extra = true
    init_typed = true
    # Do NOT set debug_dataclass_transform — this plugin replaces that workaround

Caveats
-------
- ``__replace__()`` uses Python field names (not aliases), matching pydantic's
  runtime behavior where ``__replace__`` delegates to ``model_copy()``.
- Bypasses Pydantic runtime validation (same as ``model_copy``). Pydantic's
  ``extra='forbid'`` and ``strict=True`` still validate at runtime.
- Depends on ``PydanticPlugin``, ``PydanticModelField``, and ``METADATA_KEY``
  from ``pydantic.mypy``. These are stable across pydantic 2.x but are not
  part of pydantic's public API contract.
- If pydantic adds native ``__replace__`` synthesis, the ``plugin_generated``
  guard makes this a no-op — the plugin remains safe to keep installed.

Upstream context
----------------
- pydantic/pydantic#10573 — Added ``__replace__`` runtime support (v2.10)
- pydantic/pydantic#10979 — Hid ``__replace__`` behind ``if not TYPE_CHECKING``
  so type checkers synthesize it via ``@dataclass_transform()``
- pydantic/pydantic#11010 — Confirmed ``__replace__`` uses Python names, not aliases
- pydantic/pydantic#10168 — ``populate_by_name`` + aliases: exponential overloads
  make full ``__init__`` support infeasible (closed won't-fix)
- python/mypy#17471, #18216 — mypy ``__replace__`` support for dataclass_transform
  (landed in mypy 1.11, refined in 1.15)
- PEP 681 — Data Class Transforms specification
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mypy.nodes import TypeInfo
from mypy.plugin import ClassDefContext, Plugin
from mypy.plugins.common import add_method_to_class
from mypy.typevars import fill_typevars
from pydantic.mypy import METADATA_KEY, PydanticModelField, PydanticPlugin


def plugin(version: str) -> type[Plugin]:
    """Mypy plugin entry point."""
    return PydanticReplacePlugin


class PydanticReplacePlugin(PydanticPlugin):
    """Extends PydanticPlugin with ``__replace__()`` synthesis on model classes."""

    def get_base_class_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        """Chain: pydantic transform -> __replace__ synthesis."""
        pydantic_hook = super().get_base_class_hook(fullname)
        if pydantic_hook is None:
            return None

        def chained_hook(ctx: ClassDefContext) -> None:
            pydantic_hook(ctx)
            _synthesize_replace(ctx)

        return chained_hook


def _collect_parent_fields(info: TypeInfo) -> dict[str, Any]:
    """Collect field data from the first parent that defines each field.

    When a subclass narrows a field type (e.g., ``name: str`` to
    ``name: Literal['LCP']``), ``__replace__`` must use the parent's wider
    type to satisfy LSP. Returns a mapping of field name to the parent's
    serialized field data for any field found in a parent's pydantic metadata.
    """
    parent_fields: dict[str, Any] = {}
    for parent in info.mro[1:]:
        parent_meta = parent.metadata.get(METADATA_KEY)
        if parent_meta is None:
            continue
        for name, data in parent_meta.get('fields', {}).items():
            if name not in parent_fields:
                parent_fields[name] = data
    return parent_fields


def _synthesize_replace(ctx: ClassDefContext) -> None:
    """Add a typed ``__replace__`` method to a Pydantic model class.

    Reads field metadata stored by the pydantic transformer and builds
    keyword-only optional arguments matching ``__init__`` fields (using
    Python names, not aliases).
    """
    if ctx.api.options.python_version < (3, 13):
        return

    info = ctx.cls.info

    # Don't overwrite an explicitly defined or previously synthesized __replace__
    existing = info.names.get('__replace__')
    if existing is not None and existing.plugin_generated:
        return

    metadata = info.metadata.get(METADATA_KEY)
    if metadata is None:
        return

    fields_data: dict[str, Any] = metadata.get('fields', {})
    config: dict[str, Any] = metadata.get('config', {})
    if not fields_data:
        return

    # For inherited fields that narrow a parent type (e.g., name: str ->
    # name: Literal['LCP']), use the parent's wider type for __replace__
    # args to satisfy Liskov substitution.
    parent_fields = _collect_parent_fields(info)

    model_strict = bool(config.get('strict', False))
    typed = True  # __replace__ args are always typed for maximum safety

    args = []
    for name, data in fields_data.items():
        field_data = parent_fields.get(name, data)
        field = PydanticModelField.deserialize(info, field_data, ctx.api)
        arg = field.to_argument(
            current_info=info,
            typed=typed,
            model_strict=model_strict,
            force_optional=True,  # All __replace__ args are optional
            use_alias=False,  # Python names, matching runtime behavior
            api=ctx.api,
            force_typevars_invariant=False,
            is_root_model_root=False,
        )
        args.append(arg)

    add_method_to_class(
        ctx.api,
        ctx.cls,
        '__replace__',
        args=args,
        return_type=fill_typevars(info),
    )
