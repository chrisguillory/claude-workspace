from __future__ import annotations

import os
import stat

from claude_remote_bash.auth import config_path, load_config
from claude_remote_bash.diagnose.types import Status, VectorResult

__all__ = [
    'check_config',
]


def check_config() -> VectorResult:
    """Verify the daemon config exists, has both fields populated, and is mode 0600."""
    path = config_path()
    config = load_config()

    if config is None:
        return VectorResult(
            name='config',
            status=Status.FAIL,
            summary=f'no config at {path}',
            detail='',
            fix_suggestion='claude-remote-bash-daemon init  (or `join <key>` to join an existing mesh)',
        )

    issues: list[str] = []
    fix = ''
    if not config.auth_key:
        issues.append('auth_key is empty')
        fix = 'claude-remote-bash-daemon init  (regenerate auth key)'
    if not config.name:
        issues.append('name (alias) is empty')
        if not fix:
            fix = 'claude-remote-bash-daemon set-name <alias>'

    st = path.stat()
    mode = stat.S_IMODE(st.st_mode)
    if mode != 0o600:
        issues.append(f'permissions are {mode:04o}, expected 0600')
        if not fix:
            fix = f'chmod 600 {path}'
    if st.st_uid != os.geteuid():
        issues.append(f'owned by uid={st.st_uid}, expected {os.geteuid()}')

    if issues:
        return VectorResult(
            name='config',
            status=Status.FAIL,
            summary=f'{len(issues)} problem(s) with {path}',
            detail='\n'.join(issues),
            fix_suggestion=fix,
        )

    return VectorResult(
        name='config',
        status=Status.OK,
        summary=f'{path} valid (name={config.name!r}, mode 0600)',
        detail='',
        fix_suggestion='',
    )
