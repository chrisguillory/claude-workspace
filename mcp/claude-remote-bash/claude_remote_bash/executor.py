"""Stateless command execution with marker-based output capture."""

from __future__ import annotations

import asyncio
import os
import signal
import uuid

__all__ = [
    'CommandResult',
    'execute_command',
]


class CommandResult:
    """Result of a single command execution.

    The marker used to detect end-of-command lands on stdout — the EXIT
    trap's ``echo`` writes there — so it can be parsed out cleanly without
    affecting stderr.
    """

    def __init__(self, *, stdout: str, stderr: str, exit_code: int, cwd: str) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.cwd = cwd


async def execute_command(
    command: str,
    *,
    cwd: str | None = None,
    shell: str = '/bin/zsh',
    timeout: float = 120.0,
) -> CommandResult:
    """Execute a command in a fresh login shell and capture the result.

    The command is wrapped with a unique end marker that encodes the exit code
    and post-command working directory. The marker is emitted to stdout by an
    EXIT trap, so stderr is returned verbatim.

    Args:
        command: Shell command to execute.
        cwd: Working directory. Defaults to $HOME.
        shell: Shell binary. Defaults to /bin/zsh.
        timeout: Seconds before SIGKILL. Defaults to 120.

    Returns:
        CommandResult with stdout, stderr, exit_code, and cwd.
    """
    effective_cwd = cwd or os.path.expanduser('~')
    marker = f'__CRBD_{uuid.uuid4().hex}__'

    # The trap captures $? into __rc BEFORE running the leading bare
    # ``echo`` — otherwise echo's own success clobbers $? to 0. The
    # leading echo ensures the marker lands on its own line even when
    # the command's stdout has no trailing newline (e.g. ``cat`` of a
    # file with no final \n); the parser also tolerates an embedded
    # marker as a defense-in-depth measure.
    # EXIT trap fires even if the command calls ``exit``; $(pwd -P)
    # captures the post-command working directory.
    wrapped = (
        f'trap \'__rc=$?; echo; echo "{marker}_${{__rc}}_$(pwd -P)"\' EXIT; '
        f'cd {_shell_quote(effective_cwd)} && {command}'
    )

    process = await asyncio.create_subprocess_exec(
        shell,
        '-l',
        '-c',
        wrapped,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, 'PS1': '', 'TERM': 'dumb'},
        limit=1024 * 1024,  # 1 MB per-stream buffer
        # Run the shell + its descendants in a dedicated process group so a
        # single killpg cleans up the whole tree (the user's command and any
        # children) rather than just the shell wrapper.
        start_new_session=True,
    )

    try:
        raw_stdout, raw_stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        _kill_process_group(process)
        await process.wait()
        return CommandResult(stdout='[TIMEOUT]', stderr='', exit_code=-1, cwd=effective_cwd)
    except asyncio.CancelledError:
        # Daemon shutdown cancels the handler task awaiting us; kill the whole
        # process group so the user's command (and any descendants) don't
        # survive as orphans reparented to launchd.
        _kill_process_group(process)
        await process.wait()
        raise

    stdout_text = raw_stdout.decode(errors='replace')
    stderr_text = raw_stderr.decode(errors='replace').rstrip('\n')
    return _parse_output(stdout_text, stderr_text, marker, effective_cwd)


def _parse_output(stdout_text: str, stderr_text: str, marker: str, fallback_cwd: str) -> CommandResult:
    """Extract stdout, stderr, exit code, and CWD from marker-delimited output.

    Uses ``rfind`` to locate the marker — this tolerates a marker that's
    embedded mid-line (which happens when the command's stdout has no
    trailing newline before the trap's echo). Everything before the marker
    position is the command's stdout; everything from the marker up to the
    next ``\\n`` (or end-of-stream) is the marker line.
    """
    marker_prefix = f'{marker}_'
    idx = stdout_text.rfind(marker_prefix)

    if idx == -1:
        # No marker found — process likely crashed before the trap fired.
        return CommandResult(stdout=stdout_text.rstrip('\n'), stderr=stderr_text, exit_code=-1, cwd=fallback_cwd)

    pre = stdout_text[:idx]
    end = stdout_text.find('\n', idx)
    marker_line = stdout_text[idx:] if end == -1 else stdout_text[idx:end]

    # Parse: __CRBD_<hex>__<exit_code>_<cwd>
    remainder = marker_line[len(marker_prefix) :]
    parts = remainder.split('_', 1)
    if len(parts) == 2:
        exit_code = int(parts[0])
        cwd = parts[1]
    else:
        exit_code = int(parts[0]) if parts[0] else -1
        cwd = fallback_cwd

    return CommandResult(stdout=pre.rstrip('\n'), stderr=stderr_text, exit_code=exit_code, cwd=cwd)


def _shell_quote(s: str) -> str:
    """Single-quote a string for shell use, handling embedded single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    """SIGKILL the subprocess's process group. Requires ``start_new_session=True`` at spawn."""
    if process.pid is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
