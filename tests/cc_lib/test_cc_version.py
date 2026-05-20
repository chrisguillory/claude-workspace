"""Tests for ``cc_lib.types.CCVersion``.

Covers strict construction, the ``parse`` suffix-stripping convenience,
PEP 440 version ordering, hashing, and Pydantic field round-trip.
"""

from __future__ import annotations

import pydantic
import pytest
from cc_lib.types import CCVersion
from packaging.version import InvalidVersion


class TestConstruction:
    def test_direct_construction_strict(self) -> None:
        v = CCVersion('2.1.131')
        assert v.major == 2
        assert v.minor == 1
        assert v.micro == 131

    def test_direct_construction_rejects_suffix(self) -> None:
        with pytest.raises(InvalidVersion):
            CCVersion('2.1.131 (Claude Code)')

    def test_direct_construction_rejects_garbage(self) -> None:
        with pytest.raises(InvalidVersion):
            CCVersion('not a version')


class TestParse:
    def test_parse_plain_version(self) -> None:
        assert CCVersion.parse('2.1.131') == CCVersion('2.1.131')

    def test_parse_strips_claude_suffix(self) -> None:
        assert CCVersion.parse('2.1.131 (Claude Code)') == CCVersion('2.1.131')

    def test_parse_strips_surrounding_whitespace(self) -> None:
        assert CCVersion.parse('  2.1.131 (Claude Code)\n') == CCVersion('2.1.131')

    def test_parse_rejects_empty(self) -> None:
        with pytest.raises(InvalidVersion):
            CCVersion.parse('')

    def test_parse_rejects_whitespace_only(self) -> None:
        with pytest.raises(InvalidVersion):
            CCVersion.parse('   \n')


class TestOrdering:
    def test_pep_440_numeric_ordering(self) -> None:
        """Dotted-numeric ordering, not lexicographic."""
        assert CCVersion('2.1.10') > CCVersion('2.1.9')
        assert CCVersion('2.1.100') > CCVersion('2.1.99')

    def test_equality(self) -> None:
        assert CCVersion('2.1.131') == CCVersion('2.1.131')

    def test_hash_consistent_with_equality(self) -> None:
        assert hash(CCVersion('2.1.131')) == hash(CCVersion('2.1.131'))
        assert len({CCVersion('2.1.131'), CCVersion('2.1.131')}) == 1


class TestPydanticRoundTrip:
    def test_str_input_validates(self) -> None:
        m = _Model.model_validate({'version': '2.1.131'})
        assert isinstance(m.version, CCVersion)
        assert m.version == CCVersion('2.1.131')

    def test_str_input_with_suffix_validates(self) -> None:
        m = _Model.model_validate({'version': '2.1.131 (Claude Code)'})
        assert m.version == CCVersion('2.1.131')

    def test_instance_passes_through(self) -> None:
        v = CCVersion('2.1.131')
        m = _Model.model_validate({'version': v})
        assert m.version is v

    def test_none_passes_through(self) -> None:
        m = _Model.model_validate({'version': None})
        assert m.version is None

    def test_python_mode_preserves_instance(self) -> None:
        m = _Model(version=CCVersion('2.1.131'))
        dumped = m.model_dump()
        assert isinstance(dumped['version'], CCVersion)
        assert dumped['version'] == CCVersion('2.1.131')

    def test_json_mode_serializes_as_string(self) -> None:
        m = _Model(version=CCVersion('2.1.131'))
        assert m.model_dump(mode='json') == {'version': '2.1.131'}
        assert m.model_dump_json() == '{"version":"2.1.131"}'

    def test_json_round_trip(self) -> None:
        m = _Model(version=CCVersion('2.1.131'))
        restored = _Model.model_validate_json(m.model_dump_json())
        assert restored.version == m.version
        assert isinstance(restored.version, CCVersion)

    def test_non_string_non_version_input_raises_validation_error(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _Model.model_validate({'version': 12345})

    def test_invalid_version_string_raises_validation_error(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _Model.model_validate({'version': 'not a version'})


class TestJsonSchemaGeneration:
    def test_optional_field_produces_string_schema(self) -> None:
        schema = _Model.model_json_schema()
        field = schema['properties']['version']
        # CCVersion | None → anyOf [string, null]
        assert any(branch.get('type') == 'string' for branch in field['anyOf'])

    def test_required_field_produces_string_schema(self) -> None:
        class Required(pydantic.BaseModel):
            version: CCVersion

        field = Required.model_json_schema()['properties']['version']
        assert field['type'] == 'string'


class _Model(pydantic.BaseModel):
    version: CCVersion | None = None
