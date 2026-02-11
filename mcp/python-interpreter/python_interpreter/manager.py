"""External Python interpreter subprocess manager."""

from __future__ import annotations

__all__ = [
    'ExternalInterpreterManager',
    'InterpreterConfig',
    'ExternalInterpreterError',
]

import contextlib
import dataclasses
import datetime
import importlib.resources
import json
import os
import pathlib
import select
import subprocess
from typing import Any

from python_interpreter.models import (
    DriverExecuteResponse,
    DriverListVarsResponse,
    DriverReadyResponse,
    DriverResetResponse,
)


@dataclasses.dataclass(frozen=True, slots=True)
class InterpreterConfig:
    """Configuration for an external Python interpreter."""

    name: str
    python_path: pathlib.Path
    cwd: pathlib.Path | None = None
    env: dict[str, str] | None = None
    startup_script: str | None = None


class ExternalInterpreterError(Exception):
    """External interpreter operation failed."""

    pass


class ExternalInterpreterManager:
    """Manages external Python interpreter subprocesses."""

    def __init__(self, project_dir: pathlib.Path) -> None:
        self.project_dir = project_dir
        self._interpreters: dict[str, tuple[InterpreterConfig, subprocess.Popen[str], datetime.datetime]] = {}
        self._driver_script = importlib.resources.files('python_interpreter').joinpath('driver.py').read_text()

    def add_interpreter(self, config: InterpreterConfig) -> tuple[int, datetime.datetime]:
        """Start an external interpreter subprocess.

        Returns:
            (pid, started_at) tuple
        """
        if config.name in self._interpreters:
            raise ExternalInterpreterError(f"Interpreter '{config.name}' already exists")

        env = os.environ.copy()
        if config.env:
            env.update(config.env)

        cwd = config.cwd or self.project_dir

        proc = subprocess.Popen(
            [str(config.python_path), '-c', self._driver_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(cwd),
            env=env,
            bufsize=1,
        )

        started_at = datetime.datetime.now(datetime.UTC)

        # Wait for ready signal
        response_dict = self._read_response(proc, timeout=10.0)
        try:
            ready_response = DriverReadyResponse.model_validate(response_dict)
        except Exception as e:
            self._kill_subprocess(proc)
            raise ExternalInterpreterError(f'Invalid ready response: {e}') from e

        if ready_response.status != 'ready':
            self._kill_subprocess(proc)
            raise ExternalInterpreterError(f'Failed to start: {response_dict}')

        # Execute startup script if provided
        if config.startup_script:
            result = self._send_request(proc, {'action': 'execute', 'code': config.startup_script}, timeout=30.0)
            if result.get('error'):
                self._kill_subprocess(proc)
                raise ExternalInterpreterError(f'Startup script failed:\n{result["error"]}')

        self._interpreters[config.name] = (config, proc, started_at)
        return proc.pid, started_at

    def stop_interpreter(self, name: str) -> None:
        """Stop and remove an external interpreter."""
        if name == 'builtin':
            raise ExternalInterpreterError("Cannot stop 'builtin' interpreter")

        if name not in self._interpreters:
            raise ExternalInterpreterError(f"Interpreter '{name}' not found")

        _, proc, _ = self._interpreters.pop(name)
        self._kill_subprocess(proc)

    def get_interpreters(self) -> list[tuple[str, InterpreterConfig, int, datetime.datetime]]:
        """Get all running interpreters.

        Returns:
            List of (name, config, pid, started_at) tuples. Dead interpreters are removed.
        """
        result = []
        dead = []

        for name, (config, proc, started_at) in self._interpreters.items():
            if proc.poll() is not None:
                dead.append(name)
                continue
            result.append((name, config, proc.pid, started_at))

        for name in dead:
            del self._interpreters[name]

        return result

    def execute(self, name: str, code: str, timeout: float = 60.0) -> DriverExecuteResponse:
        """Execute code in named interpreter."""
        if name not in self._interpreters:
            raise ExternalInterpreterError(f"Interpreter '{name}' not found")

        _, proc, _ = self._interpreters[name]

        if proc.poll() is not None:
            del self._interpreters[name]
            raise ExternalInterpreterError(f"Interpreter '{name}' crashed (exit code: {proc.returncode})")

        response_dict = self._send_request(proc, {'action': 'execute', 'code': code}, timeout=timeout)
        return DriverExecuteResponse.model_validate(response_dict)

    def list_vars(self, name: str, timeout: float = 10.0) -> DriverListVarsResponse:
        """List variables in named interpreter."""
        if name not in self._interpreters:
            raise ExternalInterpreterError(f"Interpreter '{name}' not found")

        _, proc, _ = self._interpreters[name]

        if proc.poll() is not None:
            del self._interpreters[name]
            raise ExternalInterpreterError(f"Interpreter '{name}' crashed")

        response_dict = self._send_request(proc, {'action': 'list_vars'}, timeout=timeout)
        return DriverListVarsResponse.model_validate(response_dict)

    def reset(self, name: str, timeout: float = 10.0) -> DriverResetResponse:
        """Reset interpreter scope (clear all variables)."""
        if name not in self._interpreters:
            raise ExternalInterpreterError(f"Interpreter '{name}' not found")

        _, proc, _ = self._interpreters[name]

        if proc.poll() is not None:
            del self._interpreters[name]
            raise ExternalInterpreterError(f"Interpreter '{name}' crashed")

        response_dict = self._send_request(proc, {'action': 'reset'}, timeout=timeout)
        return DriverResetResponse.model_validate(response_dict)

    def shutdown_all(self) -> None:
        """Kill all interpreter subprocesses. Best effort."""
        for name, (_, proc, _) in list(self._interpreters.items()):
            self._kill_subprocess(proc)
        self._interpreters.clear()

    def _send_request(self, proc: subprocess.Popen[str], request: dict[str, Any], timeout: float) -> dict[str, Any]:
        """Send request and read response."""
        if not proc.stdin or not proc.stdout:
            raise ExternalInterpreterError('Subprocess stdin/stdout not available')

        json_data = json.dumps(request)
        proc.stdin.write(f'{len(json_data)}\n{json_data}')
        proc.stdin.flush()

        return self._read_response(proc, timeout)

    def _read_response(self, proc: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
        """Read length-prefixed JSON response with timeout.

        Uses select.select to guard the initial readline() which blocks until
        the length prefix arrives. The subsequent read(length) does NOT use select
        because readline() with buffered TextIOWrapper may have already consumed
        the payload into userspace buffer — select only sees kernel-level fd
        readability and would falsely timeout on already-buffered data.
        """
        if not proc.stdout:
            raise ExternalInterpreterError('Subprocess stdout not available')

        # Guard length prefix read with select (blocks until data arrives)
        ready, _, _ = select.select([proc.stdout], [], [], timeout)
        if not ready:
            raise ExternalInterpreterError(f'Timeout ({timeout}s) waiting for response')

        length_line = proc.stdout.readline()
        if not length_line:
            raise ExternalInterpreterError('Subprocess closed stdout')

        length = int(length_line.strip())

        # Read payload — no select guard here; see docstring
        json_data = proc.stdout.read(length)

        if len(json_data) < length:
            raise ExternalInterpreterError(f'Incomplete response: expected {length}, got {len(json_data)}')

        parsed: dict[str, Any] = json.loads(json_data)
        return parsed

    def _kill_subprocess(self, proc: subprocess.Popen[str]) -> None:
        """Kill subprocess."""
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=5.0)
