from __future__ import annotations

__all__ = [
    'configure_logging',
]

import logging
import sys
from collections.abc import Mapping

from cc_lib.settings_env import get_cc_env_var


def configure_logging() -> None:
    """Configure root logging for stderr + per-module overrides from ``CC_LOG``.

    ``CC_LOG`` is a comma-separated rule list (``RUST_LOG``-subset grammar):
    a bare token sets the root level; ``logger.name=level`` sets that logger's
    level. Children inherit via Python's hierarchy. Default root is ``INFO``.

    Examples::

        CC_LOG = warning, cc_lib.mcp.bridge = debug  # bridge DEBUG, rest WARNING
        CC_LOG = warning, cc_lib.mcp = debug  # all cc_lib.mcp.* at DEBUG
        CC_LOG = info, uvicorn.access = warning  # quiet just access logs
        CC_LOG = debug  # firehose
    """
    root_level, per_logger = _parse_cc_log(get_cc_env_var('CC_LOG') or '')
    logging.basicConfig(
        level=root_level or 'INFO',
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
        force=True,
    )
    for name, level in per_logger.items():
        logging.getLogger(name).setLevel(level)


def _parse_cc_log(spec: str) -> tuple[str | None, Mapping[str, str]]:
    root: str | None = None
    per_logger: dict[str, str] = {}
    for rule in spec.split(','):
        rule = rule.strip()
        if not rule:
            continue
        if '=' in rule:
            name, _, level = rule.partition('=')
            per_logger[name.strip()] = level.strip().upper()
        else:
            root = rule.upper()
    return root, per_logger
