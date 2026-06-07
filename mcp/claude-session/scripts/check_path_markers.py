#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "claude-session",
#   "pydantic>=2.0.0",
# ]
#
# [tool.uv.sources]
# claude-session = { path = "../", editable = true }
# ///

"""Check PathMarker coverage on session models.

Path translation during cross-machine restore is driven by ``PathMarker``
annotations (``PathField``/``PathListField`` — see ``schemas/session/markers.py``);
``introspection.get_path_fields`` is the runtime consumer. This script walks every
model in ``schemas/session/models.py`` and flags suspect-named, string-typed fields
that carry no marker: each is either a translation gap (mark it) or a judged
non-path (add it to ``EXEMPT`` with the reason).

Usage:
    ./scripts/check_path_markers.py    # exit 1 while unadjudicated suspects remain
"""

from __future__ import annotations

import inspect
import re
import sys
import typing
from collections.abc import Mapping

from claude_session.introspection import get_path_fields
from claude_session.schemas.session import models
from pydantic import BaseModel

SUSPECT_NAME = re.compile(r'path|cwd|dir(?!ection)|folder|file', re.IGNORECASE)

EXEMPT: Mapping[tuple[str, str], str] = {}
"""(model, field) → reason a suspect-named string field is judged not a translatable path."""


def main() -> int:
    suspects: list[tuple[str, str, str]] = []
    n_models = 0
    n_marked = 0
    for name, cls in inspect.getmembers(models, inspect.isclass):
        if not (issubclass(cls, BaseModel) and cls.__module__ == models.__name__):
            continue
        n_models += 1
        marked = set(get_path_fields(cls))
        n_marked += len(marked)
        for field_name, info in cls.model_fields.items():
            if field_name in marked or (name, field_name) in EXEMPT:
                continue
            if SUSPECT_NAME.search(field_name) and _contains_str(info.annotation):
                suspects.append((name, field_name, str(info.annotation)))

    print(f'{n_models} models; {n_marked} PathMarker-covered fields; {len(EXEMPT)} exemptions')
    if not suspects:
        print('No unadjudicated suspects — coverage clean.')
        return 0
    print(f'\n{len(suspects)} suspect fields (each is a translation gap, or belongs in EXEMPT with a reason):')
    for model_name, field_name, annotation in sorted(suspects):
        print(f'  {model_name}.{field_name}: {annotation}')
    return 1


def _contains_str(annotation: object) -> bool:
    """Whether the annotation's atoms include ``str`` (through aliases, unions, containers)."""
    if isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    if annotation is str:
        return True
    return any(_contains_str(arg) for arg in typing.get_args(annotation))


if __name__ == '__main__':
    sys.exit(main())
