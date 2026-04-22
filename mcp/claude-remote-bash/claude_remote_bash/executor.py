"""Stateless command execution with marker-based output capture."""

from __future__ import annotations

import asyncio
import os
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

    # EXIT trap guarantees the marker prints even if the command calls `exit`.
    # The trap captures $? (the exit code) and $(pwd -P) (the working directory).
    wrapped = f'trap \'echo "{marker}_$?_$(pwd -P)"\' EXIT; cd {_shell_quote(effective_cwd)} && {command}'

    process = await asyncio.create_subprocess_exec(
        shell,
        '-l',
        '-c',
        wrapped,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, 'PS1': '', 'TERM': 'dumb'},
        limit=1024 * 1024,  # 1 MB per-stream buffer
    )

    try:
        raw_stdout, raw_stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout,
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return CommandResult(stdout='[TIMEOUT]', stderr='', exit_code=-1, cwd=effective_cwd)

    stdout_text = raw_stdout.decode(errors='replace')
    stderr_text = raw_stderr.decode(errors='replace').rstrip('\n')
    return _parse_output(stdout_text, stderr_text, marker, effective_cwd)


def _parse_output(stdout_text: str, stderr_text: str, marker: str, fallback_cwd: str) -> CommandResult:
    """Extract stdout, stderr, exit code, and CWD from marker-delimited output.

    The marker line is carved out of stdout; the command's own stdout is
    everything before it. stderr is passed through untouched since the
    marker never lands there.
    """
    marker_prefix = f'{marker}_'
    lines = stdout_text.split('\n')

    # Find the marker line (search from the end — it's the last line before exit)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith(marker_prefix):
            marker_line = lines[i]
            stdout = '\n'.join(lines[:i]).rstrip('\n')

            # Parse: __CRBD_<hex>__<exit_code>_<cwd>
            remainder = marker_line[len(marker_prefix) :]
            parts = remainder.split('_', 1)
            if len(parts) == 2:
                exit_code = int(parts[0])
                cwd = parts[1]
            else:
                exit_code = int(parts[0]) if parts[0] else -1
                cwd = fallback_cwd

            return CommandResult(stdout=stdout, stderr=stderr_text, exit_code=exit_code, cwd=cwd)

    # No marker found — process likely crashed before reaching it
    return CommandResult(stdout=stdout_text.rstrip('\n'), stderr=stderr_text, exit_code=-1, cwd=fallback_cwd)


def _shell_quote(s: str) -> str:
    """Single-quote a string for shell use, handling embedded single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"
