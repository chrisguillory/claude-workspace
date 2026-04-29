"""Implementation of the ``CC_LITERAL_RELAX=1`` escape hatch.

Walks a Pydantic core schema and replaces ``literal_schema`` nodes with a
``str_schema`` plus an after-validator that logs novel values. Imported
by ``base.py``; not part of the public API.
"""

from __future__ import annotations

__all__ = [
    'maybe_relax_literals',
]

import logging
from collections.abc import Callable, Set

from pydantic_core import CoreSchema, core_schema

from cc_lib.settings_env import get_cc_env_var

logger = logging.getLogger(__name__)


def maybe_relax_literals(schema: CoreSchema) -> CoreSchema:
    """Apply the ``CC_LITERAL_RELAX=1`` escape hatch if set, otherwise pass through."""
    if get_cc_env_var('CC_LITERAL_RELAX') == '1':
        return _relax_literals_in_schema(schema)
    return schema


def _relax_literals_in_schema(schema: CoreSchema) -> CoreSchema:
    """Walk a Pydantic core schema and relax ``literal_schema`` nodes.

    Handles common compositions: union, nullable, default, model, model-fields,
    list, dict, and function-* wrappers. Other schema types pass through —
    extend if a schema shape comes up that isn't covered.
    """
    schema_type = schema['type']

    if schema_type == 'literal':
        allowed = frozenset(schema['expected'])
        return core_schema.no_info_after_validator_function(
            _make_relaxed_literal_validator(allowed),
            core_schema.str_schema(),
        )

    if schema_type == 'union':
        return {
            **schema,
            'choices': [
                (_relax_literals_in_schema(c[0]), c[1]) if isinstance(c, tuple) else _relax_literals_in_schema(c)
                for c in schema['choices']
            ],
        }

    if schema_type in {'nullable', 'default', 'model'}:
        return {**schema, 'schema': _relax_literals_in_schema(schema['schema'])}

    if schema_type == 'model-fields':
        return {
            **schema,
            'fields': {k: {**v, 'schema': _relax_literals_in_schema(v['schema'])} for k, v in schema['fields'].items()},
        }

    if schema_type == 'list' and 'items_schema' in schema:
        return {**schema, 'items_schema': _relax_literals_in_schema(schema['items_schema'])}

    if schema_type == 'dict' and 'values_schema' in schema:
        return {**schema, 'values_schema': _relax_literals_in_schema(schema['values_schema'])}

    if (
        schema_type
        in {
            'function-after',
            'function-before',
            'function-wrap',
            'function-plain',
        }
        and 'schema' in schema
    ):
        return {**schema, 'schema': _relax_literals_in_schema(schema['schema'])}

    return schema


def _make_relaxed_literal_validator(allowed: Set[str]) -> Callable[[str], str]:
    """Return an after-validator that logs values outside ``allowed`` but accepts them."""

    def validator(v: str) -> str:
        if v not in allowed:
            logger.info(
                'CC_LITERAL_RELAX observed novel value: %r (known: %s)',
                v,
                sorted(allowed),
            )
        return v

    return validator
