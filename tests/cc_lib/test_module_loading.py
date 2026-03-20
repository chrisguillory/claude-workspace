"""Tests for load_module_from_path and temporary_module utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
from cc_lib.utils import load_module_from_path, temporary_module


class TestLoadModuleFromPath:
    """Verify load_module_from_path loads files and sets module attributes."""

    def test_loads_module_with_correct_name(self, tmp_path: Path) -> None:
        f = tmp_path / 'my_mod.py'
        f.write_text('X = 42')
        mod = load_module_from_path(f)
        assert mod.__name__ == 'my_mod'
        assert mod.X == 42

    def test_sets_correct_file(self, tmp_path: Path) -> None:
        f = tmp_path / 'my_mod.py'
        f.write_text('pass')
        mod = load_module_from_path(f)
        assert mod.__file__ == str(f)

    def test_auto_derives_name_hyphens_to_underscores(self, tmp_path: Path) -> None:
        f = tmp_path / 'my-mod.py'
        f.write_text('Y = 99')
        mod = load_module_from_path(f)
        assert mod.__name__ == 'my_mod'
        assert mod.Y == 99

    def test_explicit_module_name_overrides(self, tmp_path: Path) -> None:
        f = tmp_path / 'my_mod.py'
        f.write_text('pass')
        mod = load_module_from_path(f, module_name='custom')
        assert mod.__name__ == 'custom'

    def test_not_registered_in_sys_modules(self, tmp_path: Path) -> None:
        f = tmp_path / 'ephemeral.py'
        f.write_text('pass')
        mod = load_module_from_path(f)
        assert mod.__name__ not in sys.modules

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match='module file not found'):
            load_module_from_path(tmp_path / 'nonexistent.py')

    def test_syntax_error_propagates(self, tmp_path: Path) -> None:
        f = tmp_path / 'bad_syntax.py'
        f.write_text('def !!!')
        with pytest.raises(SyntaxError):
            load_module_from_path(f)

    def test_runtime_error_propagates(self, tmp_path: Path) -> None:
        f = tmp_path / 'bad_init.py'
        f.write_text("raise RuntimeError('boom')")
        with pytest.raises(RuntimeError, match='boom'):
            load_module_from_path(f)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        f = tmp_path / 'str_path.py'
        f.write_text('Z = 1')
        mod = load_module_from_path(str(f))
        assert mod.Z == 1


class TestTemporaryModule:
    """Verify temporary_module registers and cleans up sys.modules."""

    def test_registers_then_cleans_up(self, tmp_path: Path) -> None:
        f = tmp_path / 'temp_mod.py'
        f.write_text('pass')
        with temporary_module(f) as mod:
            assert 'temp_mod' in sys.modules
            assert sys.modules['temp_mod'] is mod
        assert 'temp_mod' not in sys.modules

    def test_restores_prior_entry(self, tmp_path: Path) -> None:
        f = tmp_path / 'temp_mod.py'
        f.write_text('pass')
        prior = ModuleType('temp_mod')
        sys.modules['temp_mod'] = prior
        try:
            with temporary_module(f) as mod:
                assert sys.modules['temp_mod'] is mod
                assert sys.modules['temp_mod'] is not prior
            assert sys.modules['temp_mod'] is prior
        finally:
            sys.modules.pop('temp_mod', None)

    def test_cleans_up_on_exception(self, tmp_path: Path) -> None:
        f = tmp_path / 'temp_mod.py'
        f.write_text('pass')
        with pytest.raises(ValueError, match='inside block'), temporary_module(f):
            raise ValueError('inside block')
        assert 'temp_mod' not in sys.modules
