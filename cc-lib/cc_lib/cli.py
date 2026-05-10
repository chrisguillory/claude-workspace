"""Typer CLI factories for claude-workspace scripts.

Typer is a dependency of cc-lib (declared in pyproject.toml).

Provides:
    create_app                  Build a Typer app with standard conventions
    add_install_command         Register install/uninstall/completion subcommands (standalone scripts)
    add_completion_command      Register completion-only subcommands (uv tool install packages)
    add_help_command            Register a `help` subcommand for top-level / per-subcommand / recursive help
    run_app                     Entry-point helper: derives prog_name from sys.argv[0]
"""

from __future__ import annotations

__all__ = [
    'add_completion_command',
    'add_help_command',
    'add_install_command',
    'create_app',
    'run_app',
]

import difflib
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypeAliasType, get_args, get_origin

import click
import rich.console
import typer
import typer.completion
import typer.main

from cc_lib.utils.atomic_write import atomic_write


def create_app(*, help: str) -> typer.Typer:  # noqa: A002 — standard CLI parameter name, passed to typer.Typer(help=)
    """Create a Typer app with standard conventions.

    - ``add_completion=False`` (disables typer's broken ``--install-completion``)
    - Calls ``typer.completion.completion_init()`` for runtime tab completion
    - Registers a callback that prints help on bare invocation
    - Installs the PEP 695 alias patch (idempotent, lazy)
    """
    Pep695AliasPatcher().install()
    app = typer.Typer(help=help, add_completion=False)
    typer.completion.completion_init()

    @app.callback(invoke_without_command=True)
    def _default(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            print(ctx.get_help())
            raise SystemExit(0)

    return app


def add_install_command(app: typer.Typer, *, script_path: str) -> None:
    """Register install, uninstall, and completion subcommands.

    ``install`` creates a launcher in ``~/.local/bin/`` and installs shell
    completions (auto-detected from ``$SHELL``). ``uninstall`` reverses both.
    ``completion`` is a hidden command that prints the completion script
    to stdout for debugging.

    Args:
        app: The Typer app to add commands to.
        script_path: The calling script's ``__file__`` — used to resolve
            the real path for the launcher.
    """
    launcher = LauncherInstaller(Path(script_path).resolve())
    completions = CompletionInstaller()

    @app.command(rich_help_panel='Tool Management')
    def install(
        shell: str | None = typer.Option(None, '--shell', help='Shell for completions (default: auto-detect)'),
    ) -> None:
        """Install to PATH with shell completions."""
        console = rich.console.Console(stderr=True)
        prog_name = _prog_name_from_context()

        path = launcher.install(prog_name)
        console.print(f'Launcher: [bold]{path}[/bold]')
        console.print(f'  -> {launcher.script_path}')

        shell_name = shell or _detect_shell()
        if shell_name:
            dest = completions.install(prog_name, shell_name)
            if dest:
                console.print(f'Completion: [bold]{dest}[/bold]')
        elif shell is None:
            console.print('\n[dim]Shell not detected — pass --shell zsh to add completions.[/dim]')

        console.print('\nRestart shell to activate.')

    @app.command(rich_help_panel='Tool Management')
    def uninstall(
        shell: str | None = typer.Option(None, '--shell', help='Shell for completions (default: auto-detect)'),
    ) -> None:
        """Remove from PATH and shell completions."""
        console = rich.console.Console(stderr=True)
        prog_name = _prog_name_from_context()

        removed = launcher.uninstall(prog_name)
        if removed:
            console.print(f'Removed launcher: [bold]{removed}[/bold]')
        else:
            console.print(f'Launcher not installed: {launcher.path_for(prog_name)}')

        shell_name = shell or _detect_shell()
        if shell_name:
            removed = completions.uninstall(prog_name, shell_name)
            if removed:
                console.print(f'Removed completion: [bold]{removed}[/bold]')

    @app.command(hidden=True)
    def completion(
        shell: str | None = typer.Argument(None, help='Shell type (default: auto-detect)'),
    ) -> None:
        """Print shell tab completion script to stdout."""
        prog_name = _prog_name_from_context()
        shell_name = shell or _detect_shell()
        if not shell_name:
            print('Shell not detected — pass shell name as argument.', file=sys.stderr)
            raise SystemExit(1)
        click.echo(completions.script(prog_name, shell_name))


def add_completion_command(app: typer.Typer) -> None:
    """Register completion install/uninstall subcommands for installed packages.

    Unlike ``add_install_command()``, this does NOT create a launcher script.
    Use for commands already in PATH via ``uv tool install``.
    """
    completions = CompletionInstaller()

    @app.command('install-completions', rich_help_panel='Shell Completions')
    def install_completions(
        shell: str | None = typer.Option(None, '--shell', help='Shell (default: auto-detect)'),
    ) -> None:
        """Install shell tab completions."""
        console = rich.console.Console(stderr=True)
        prog_name = _prog_name_from_context()
        shell_name = shell or _detect_shell()
        if not shell_name:
            console.print('[red]Shell not detected — pass --shell zsh[/red]')
            raise SystemExit(1)
        dest = completions.install(prog_name, shell_name)
        if dest:
            console.print(f'Completion installed: [bold]{dest}[/bold]')
            console.print('\nRestart shell to activate.')
        else:
            console.print(f'[red]Shell {shell_name!r} is not supported (zsh, bash)[/red]')
            raise SystemExit(1)

    @app.command('uninstall-completions', rich_help_panel='Shell Completions')
    def uninstall_completions(
        shell: str | None = typer.Option(None, '--shell', help='Shell (default: auto-detect)'),
    ) -> None:
        """Remove shell tab completions."""
        console = rich.console.Console(stderr=True)
        prog_name = _prog_name_from_context()
        shell_name = shell or _detect_shell()
        if not shell_name:
            console.print('[red]Shell not detected — pass --shell zsh[/red]')
            raise SystemExit(1)
        removed = completions.uninstall(prog_name, shell_name)
        if removed:
            console.print(f'Removed: [bold]{removed}[/bold]')
        else:
            console.print('No completion script found.')

    @app.command(hidden=True)
    def show_completion(
        shell: str | None = typer.Argument(None, help='Shell type (default: auto-detect)'),
    ) -> None:
        """Print shell tab completion script to stdout."""
        prog_name = _prog_name_from_context()
        shell_name = shell or _detect_shell()
        if not shell_name:
            print('Shell not detected — pass shell name as argument.', file=sys.stderr)
            raise SystemExit(1)
        click.echo(completions.script(prog_name, shell_name))


def add_help_command(app: typer.Typer) -> None:
    """Register a ``help`` subcommand: top-level / per-subcommand / recursive.

    Usage::

        <cli> help                  # bare → top-level help (same as `--help`)
        <cli> help <subcmd>         # subcommand help (same as `<subcmd> --help`)
        <cli> help --recursive      # full tree: top-level + every subcommand
        <cli> help -R               # short form

    The recursive form is the one-shot artifact AI consumers use to discover
    a CLI's full surface without round-tripping through `<cli> <subcmd> --help`
    for every subcommand.

    Args:
        app: The Typer app to add the ``help`` command to.
    """

    def _complete_subcommand(incomplete: str) -> Sequence[str]:
        click_obj = typer.main.get_command(app)
        if not isinstance(click_obj, click.Group):
            return []
        return [name for name, cmd in click_obj.commands.items() if name.startswith(incomplete) and not cmd.hidden]

    @app.command('help', rich_help_panel='Documentation')
    def help_cmd(
        subcommand: str | None = typer.Argument(
            None,
            autocompletion=_complete_subcommand,
            help='Show help for a specific subcommand. If omitted, shows top-level help.',
        ),
        recursive: bool = typer.Option(
            False,
            '--recursive',
            '-R',
            help="Show top-level help plus every non-hidden subcommand's --help in one shot.",
        ),
    ) -> None:
        """Show help — bare for top-level, with subcommand for one, --recursive for all."""
        ctx = click.get_current_context()
        if subcommand and recursive:
            raise click.UsageError(
                'Cannot combine --recursive with a subcommand argument.',
                ctx=ctx,
            )

        click_obj = typer.main.get_command(app)
        info_name = ctx.find_root().info_name

        with click.Context(click_obj, info_name=info_name) as root_ctx:
            if subcommand:
                if not isinstance(click_obj, click.Group):
                    raise click.UsageError(
                        f'No subcommands available; cannot show help for {subcommand!r}.',
                        ctx=ctx,
                    )
                cmd = click_obj.get_command(root_ctx, subcommand)
                if cmd is None:
                    suggestions = difflib.get_close_matches(
                        subcommand,
                        list(click_obj.commands.keys()),
                        n=1,
                        cutoff=0.6,
                    )
                    suffix = f' Did you mean {suggestions[0]!r}?' if suggestions else ''
                    raise click.UsageError(
                        f'No such command {subcommand!r}.{suffix}',
                        ctx=ctx,
                    )
                with click.Context(cmd, info_name=subcommand, parent=root_ctx) as sub_ctx:
                    typer.echo(sub_ctx.get_help())
                return

            if recursive and isinstance(click_obj, click.Group):
                _print_recursive_help(click_obj, root_ctx)
                return

            typer.echo(root_ctx.get_help())


def run_app(app: typer.Typer) -> None:
    """Run the app with a clean prog_name derived from sys.argv[0].

    Strips the ``.py`` extension so help text and completion vars use the
    launcher name regardless of invocation method.
    """
    app(prog_name=os.path.basename(sys.argv[0]).removesuffix('.py'))


class Pep695AliasPatcher:
    """Patch Typer to recognize PEP 695 aliases (``type X = Literal[...]``).

    Typer 0.23.x raises ``RuntimeError: Type not yet supported`` when it
    encounters a PEP 695 type alias at command registration. This patcher
    teaches Typer's ``is_literal_type`` and ``literal_values`` helpers to
    unwrap aliases via ``__value__`` until exhausted.

    Lifecycle: installed lazily from ``create_app()`` so non-CLI consumers
    of ``cc_lib`` (hooks, tests, MCP server runtime) do not mutate Typer's
    global state. Idempotent across repeat ``create_app()`` calls via the
    ``MARKER`` sentinel attribute.

    Convergence: detects when upstream lands the fix via an end-to-end
    registration probe. If the probe succeeds without our patch, raises
    ``RuntimeError`` with a "remove this" message. See fastapi/typer#970.

    Patches ``typer.main`` (not ``typer._typing``) because ``typer.main``
    does ``from ._typing import is_literal_type, literal_values`` at import
    time, capturing the names into its own namespace; patching only
    ``typer._typing`` is insufficient.

    Coverage: only ``type X = Literal[...]`` is unwrapped. Use ``X | None``
    at the call site, not inside the alias — ``type X = Literal[...] | None``
    fails with the same ``RuntimeError: Type not yet supported`` because
    ``Optional[Literal[...]]`` is a Union after unwrap, not a Literal, and
    Typer's outer Union handling can't see inside the alias.
    """

    MARKER = '__cc_lib_pep695_patch__'
    type ProbeAlias = Literal['__cc_lib_probe__']

    def install(self) -> None:
        """Install the patch (idempotent). Raises if upstream has fixed PEP 695."""
        if getattr(typer.main.is_literal_type, self.MARKER, False):  # type: ignore[attr-defined]  # Typer runtime attr
            return  # already installed

        if self._vanilla_typer_handles_pep695():
            raise RuntimeError(
                'Typer now handles PEP 695 aliases natively (registration smoke '
                'test passed without our patch). Remove Pep695AliasPatcher().install() '
                'from cc_lib.cli.create_app() — see fastapi/typer#970.'
            )

        self._apply()

    def _vanilla_typer_handles_pep695(self) -> bool:
        """End-to-end probe: register a command using a PEP 695 alias.

        Returns True iff Typer registers it without RuntimeError — catches
        any plausible shape of the upstream fix, not just one specific
        implementation.
        """
        probe_app = typer.Typer()

        @probe_app.command()
        def _probe(value: Pep695AliasPatcher.ProbeAlias = '__cc_lib_probe__') -> None:  # noqa: ARG001 — typer needs a callback
            pass

        try:
            typer.main.get_command(probe_app)
        except RuntimeError:
            return False
        return True

    def _apply(self) -> None:
        """Install the patched functions onto ``typer.main``.

        ``Any`` annotations below are intentional: Typer hands the patched
        functions arbitrary user annotations at command registration; there
        is no narrower type that's both correct and useful.
        """

        def _resolve(t: Any) -> Any:
            # isinstance(TypeAliasType) matches PR #970's actual approach;
            # avoids unwrapping non-PEP-695 user classes that happen to have __value__.
            # Depth cap guards against mutually-recursive aliases (`type A = B; type B = A`)
            # which would otherwise hang at module-load time inside create_app().
            for _ in range(32):
                if not isinstance(t, TypeAliasType):
                    return t
                t = t.__value__
            raise TypeError(f'PEP 695 alias chain exceeds depth 32 (possible recursion): {t!r}')

        def patched_is_literal_type(t: Any) -> bool:  # strict_typing_linter.py: loose-typing — Typer types
            return get_origin(_resolve(t)) is Literal

        def patched_literal_values(t: Any) -> tuple[Any, ...]:  # strict_typing_linter.py: loose-typing — Typer types
            return get_args(_resolve(t))

        setattr(patched_is_literal_type, self.MARKER, True)
        setattr(patched_literal_values, self.MARKER, True)
        typer.main.is_literal_type = patched_is_literal_type  # type: ignore[attr-defined,assignment]  # Typer runtime attr
        typer.main.literal_values = patched_literal_values  # type: ignore[attr-defined,assignment]  # Typer runtime attr


class LauncherInstaller:
    """Manages shell wrapper scripts in ``~/.local/bin/``.

    uv doesn't resolve symlinks before computing relative paths in
    ``[tool.uv.sources]``. Symlinks from ``~/.local/bin/`` cause relative
    paths to resolve from the symlink's directory, not the script's.

    This creates launcher scripts that pass the resolved script path to
    ``uv run --script``, so ``[tool.uv.sources]`` paths resolve correctly.
    """

    BIN_DIR = Path.home() / '.local' / 'bin'
    SAFE_NAME_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$')
    TEMPLATE = """\
#!/bin/sh
# Launcher for {command_name} — resolves real script path so uv's
# relative [tool.uv.sources] paths resolve from the script's directory.
# Generated by cc_lib.cli — do not edit manually.
exec uv run --no-project --script '{script_path}' "$@"
"""

    def __init__(self, script_path: Path) -> None:
        self._script_path = script_path

    @property
    def script_path(self) -> Path:
        return self._script_path

    def install(self, prog_name: str) -> Path:
        """Create launcher script. Returns installed path."""
        self._validate_name(prog_name)
        target = self._resolve_path(prog_name)
        content = self.TEMPLATE.format(
            command_name=prog_name,
            script_path=self._shell_quote(str(self._script_path)),
        )
        atomic_write(target, content.encode(), mode=0o755)
        return target

    def uninstall(self, prog_name: str) -> Path | None:
        """Remove launcher script. Returns removed path or None."""
        self._validate_name(prog_name)
        target = self.path_for(prog_name)
        if target.exists() or target.is_symlink():
            target.unlink()
            return target
        return None

    def path_for(self, prog_name: str) -> Path:
        """Return launcher path without installing."""
        return self.BIN_DIR / prog_name

    @classmethod
    def _validate_name(cls, name: str) -> None:
        """Validate command name is safe for use as a filename."""
        if not cls.SAFE_NAME_RE.match(name) or name in ('.', '..'):
            raise SystemExit(f'Error: Invalid command name: {name!r}')

    @classmethod
    def _resolve_path(cls, name: str) -> Path:
        """Compute launcher path, verifying it stays within BIN_DIR."""
        target = (cls.BIN_DIR / name).resolve()
        try:
            target.relative_to(cls.BIN_DIR.resolve())
        except ValueError:
            raise SystemExit(f'Error: Path traversal: {name!r} resolves outside {cls.BIN_DIR}') from None
        return target

    @staticmethod
    def _shell_quote(s: str) -> str:
        """Escape for embedding in a POSIX single-quoted string."""
        return s.replace("'", "'\\''")


class CompletionInstaller:
    """Manages shell tab completion scripts.

    Typer's built-in ``install()`` hardcodes ``~/.zfunc`` and appends to
    ``~/.zshrc`` (ignoring ``ZDOTDIR``), and uses shellingham for detection
    (fails outside interactive shells). This class respects ``ZDOTDIR`` and
    ``XDG`` conventions.
    """

    OVERRIDES: Mapping[str, Callable[[str], Path]] = {
        'zsh': lambda prog: Path(os.environ.get('ZDOTDIR') or Path.home() / '.zsh') / 'completions' / f'_{prog}',
        'bash': lambda prog: (
            Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')
            / 'bash-completion'
            / 'completions'
            / prog
        ),
    }

    def install(self, prog_name: str, shell: str) -> Path | None:
        """Install completion script. Returns dest path or None."""
        resolver = self.OVERRIDES.get(shell)
        if resolver is None:
            return None
        dest = resolver(prog_name)
        atomic_write(dest, self.script(prog_name, shell).encode())
        return dest

    def uninstall(self, prog_name: str, shell: str) -> Path | None:
        """Remove completion script. Returns removed path or None."""
        resolver = self.OVERRIDES.get(shell)
        if resolver is None:
            return None
        dest = resolver(prog_name)
        if dest.exists():
            dest.unlink()
            return dest
        return None

    @staticmethod
    def script(prog_name: str, shell: str) -> str:
        """Generate completion script content."""
        complete_var = f'_{prog_name.replace("-", "_").upper()}_COMPLETE'
        return typer.completion.get_completion_script(
            prog_name=prog_name,
            complete_var=complete_var,
            shell=shell,
        )


# -- Private helpers ----------------------------------------------------------


def _detect_shell() -> str | None:
    """Auto-detect shell from $SHELL environment variable."""
    name = Path(os.environ.get('SHELL', '')).name
    return name if name in ('zsh', 'bash') else None


def _prog_name_from_context() -> str:
    """Derive prog_name from the current click context."""
    ctx = click.get_current_context()
    return ctx.find_root().info_name or 'unknown'


def _print_recursive_help(group: click.Group, root_ctx: click.Context) -> None:
    """Print top-level help, then `<sub> --help` for every non-hidden subcommand.

    Hidden subcommands are skipped to match what `--help` displays. Nested
    subgroups are not currently expanded (no workspace CLI has them today).
    """
    typer.echo(root_ctx.get_help())
    for name in sorted(group.commands.keys()):
        cmd = group.commands[name]
        if getattr(cmd, 'hidden', False):
            continue
        typer.echo()
        typer.echo(f'━━━ {root_ctx.info_name} {name} ━━━')
        typer.echo()
        with click.Context(cmd, info_name=name, parent=root_ctx) as sub_ctx:
            typer.echo(sub_ctx.get_help())
