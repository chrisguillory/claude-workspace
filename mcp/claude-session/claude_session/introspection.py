"""Introspection utilities for Pydantic models.

Enables automated metadata extraction for path translation, reserved fields,
and schema evolution tracking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypedDict, Union, get_args, get_origin, overload

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from claude_session.schemas.session.markers import CCVersionMarker, PathMarker

__all__ = [
    'ModelSummary',
    'get_cc_version_fields',
    'get_field_version_info',
    'get_literal_values',
    'get_path_fields',
    'get_reserved_fields',
    'model_summary',
    'print_model_summary',
]


class ModelSummary(TypedDict):
    """Summary of a Pydantic model's metadata."""

    model_name: str
    total_fields: int
    required_fields: Sequence[str]
    optional_fields: Sequence[str]
    path_fields: Sequence[str]
    reserved_fields: Sequence[str]
    versioned_fields: Mapping[str, str]
    docstring: str | None


@overload
def get_path_fields(target: type[BaseModel]) -> Sequence[str]: ...
@overload
def get_path_fields(target: BaseModel) -> Mapping[str, str | Sequence[str]]: ...
def get_path_fields(target: BaseModel | type[BaseModel]) -> Sequence[str] | Mapping[str, str | Sequence[str]]:
    """Find all fields marked with PathMarker.

    Passing the model class returns the field names; passing a model instance
    returns a mapping of name → value for every marked field that is set
    (None values are filtered). Handles Python 3.12+ type aliases and Union
    types (e.g., ``PathField | None``).

    Example:
        >>> from claude_session.schemas.session import UserRecord
        >>> get_path_fields(UserRecord)
        ['cwd', 'projectPaths']
    """
    cls = target if isinstance(target, type) else type(target)
    names = [
        name for name, info in cls.model_fields.items() if _annotation_contains_marker(info.annotation, PathMarker)
    ]
    if isinstance(target, type):
        return names
    return {name: value for name, value in target if name in names and value is not None}


@overload
def get_cc_version_fields(target: type[BaseModel]) -> Sequence[str]: ...
@overload
def get_cc_version_fields(target: BaseModel) -> Mapping[str, str]: ...
def get_cc_version_fields(target: BaseModel | type[BaseModel]) -> Sequence[str] | Mapping[str, str]:
    """Find all fields marked with CCVersionMarker.

    Passing the model class returns the field names; passing a model instance
    returns a mapping of name → value for every marked field that is set
    (None values are filtered). Handles Python 3.12+ type aliases and Union
    types (e.g., ``CCVersionStrField | None``).

    Example:
        >>> from claude_session.schemas.session import UserRecord
        >>> get_cc_version_fields(UserRecord)
        ['version']
    """
    cls = target if isinstance(target, type) else type(target)
    names = [
        name for name, info in cls.model_fields.items() if _annotation_contains_marker(info.annotation, CCVersionMarker)
    ]
    if isinstance(target, type):
        return names
    return {name: value for name, value in target if name in names and value is not None}


def get_reserved_fields(model: type[BaseModel]) -> Mapping[str, Mapping[str, object]]:
    """Find all reserved (always-null) fields.

    Returns a mapping of field names to their metadata for fields that are
    reserved for future use and always null in current data.

    Args:
        model: Pydantic model class to inspect

    Returns:
        Dict mapping field name to json_schema_extra metadata

    Example:
        >>> from claude_session.schemas.session import UserRecord
        >>> get_reserved_fields(UserRecord)
        {'skills': {'status': 'reserved'}, 'mcp': {'status': 'reserved'}}
    """
    reserved = {}

    for field_name, field_info in model.model_fields.items():
        if (
            field_info.json_schema_extra
            and isinstance(field_info.json_schema_extra, dict)
            and field_info.json_schema_extra.get('status') == 'reserved'
        ):
            reserved[field_name] = field_info.json_schema_extra

    return reserved


def get_field_version_info(model: type[BaseModel]) -> Mapping[str, str]:
    """Get version information for fields that track when they were added.

    Args:
        model: Pydantic model class to inspect

    Returns:
        Dict mapping field name to version string

    Example:
        >>> from claude_session.schemas.session import UserRecord
        >>> get_field_version_info(UserRecord)
        {'thinkingMetadata': '2.0.35'}
    """
    version_info: dict[str, str] = {}

    for field_name, field_info in model.model_fields.items():
        if (
            field_info.json_schema_extra
            and isinstance(field_info.json_schema_extra, dict)
            and 'added_in_version' in field_info.json_schema_extra
        ):
            version = field_info.json_schema_extra['added_in_version']
            if isinstance(version, str):
                version_info[field_name] = version

    return version_info


def get_literal_values(field_info: FieldInfo) -> Sequence[str | int | bool] | None:
    """Extract literal values from a field definition.

    Args:
        field_info: Pydantic field info

    Returns:
        List of literal values if field uses Literal type, None otherwise

    Example:
        >>> from claude_session.schemas.session import UserRecord
        >>> field_info = UserRecord.model_fields['userType']
        >>> get_literal_values(field_info)
        ['external']
    """
    # This would require parsing the annotation
    # For now, return None - can be enhanced if needed
    return None


def model_summary(model: type[BaseModel]) -> ModelSummary:
    """Generate a comprehensive summary of a model's metadata.

    Args:
        model: Pydantic model class to inspect

    Returns:
        Dict with model statistics and metadata

    Example:
        >>> from claude_session.schemas.session import UserRecord
        >>> summary = model_summary(UserRecord)
        >>> summary['total_fields']
        16
        >>> summary['path_fields']
        ['cwd', 'projectPaths']
    """
    path_fields = get_path_fields(model)
    reserved = get_reserved_fields(model)
    version_info = get_field_version_info(model)

    # Count field types
    optional_fields = []
    required_fields = []

    for field_name, field_info in model.model_fields.items():
        if field_info.is_required():
            required_fields.append(field_name)
        else:
            optional_fields.append(field_name)

    return {
        'model_name': model.__name__,
        'total_fields': len(model.model_fields),
        'required_fields': required_fields,
        'optional_fields': optional_fields,
        'path_fields': path_fields,
        'reserved_fields': list(reserved.keys()),
        'versioned_fields': version_info,
        'docstring': model.__doc__,
    }


def print_model_summary(model: type[BaseModel]) -> None:
    """Print a human-readable summary of a model.

    Args:
        model: Pydantic model class to inspect
    """
    summary = model_summary(model)

    print(f'Model: {summary["model_name"]}')
    print(f'  Total fields: {summary["total_fields"]}')
    print(f'  Required: {len(summary["required_fields"])}')
    print(f'  Optional: {len(summary["optional_fields"])}')

    if summary['path_fields']:
        print(f'  Path fields: {", ".join(summary["path_fields"])}')

    if summary['reserved_fields']:
        print(f'  Reserved fields: {", ".join(summary["reserved_fields"])}')

    if summary['versioned_fields']:
        print('  Versioned fields:')
        for field, version in summary['versioned_fields'].items():
            print(f'    {field}: added in {version}')


def _annotation_contains_marker(
    annotation: Any,
    marker_class: type,
) -> bool:
    """Check if a Pydantic field annotation carries a marker instance.

    Handles direct ``Annotated[T, Marker()]``, PEP 695 type aliases
    (``type X = Annotated[...]``), and unions like ``X | None``.
    """
    if get_origin(annotation) is not None:
        args = get_args(annotation)
        for arg in args[1:]:
            if isinstance(arg, marker_class):
                return True

    if hasattr(annotation, '__value__'):
        actual_type = annotation.__value__
        if get_origin(actual_type) is not None:
            args = get_args(actual_type)
            for arg in args[1:]:
                if isinstance(arg, marker_class):
                    return True

    origin = get_origin(annotation)
    if origin is Union or (origin is not None and getattr(origin, '__name__', '') == 'UnionType'):
        for union_arg in get_args(annotation):
            if _annotation_contains_marker(union_arg, marker_class):
                return True

    return False


if __name__ == '__main__':
    # Demo
    from claude_session.schemas.session import AssistantRecord, Message, UserRecord

    print('=' * 80)
    print('Model Introspection Demo')
    print('=' * 80)
    print()

    models: Sequence[type[BaseModel]] = [UserRecord, AssistantRecord, Message]
    for model in models:
        print_model_summary(model)
        print()
