from __future__ import annotations

from collections.abc import Callable, Mapping

from claude_remote_bash.diagnose.types import DiagnoseReport, VectorResult
from claude_remote_bash.diagnose.vectors.config import check_config
from claude_remote_bash.diagnose.vectors.interpreter import check_interpreter
from claude_remote_bash.diagnose.vectors.local_network import check_local_network
from claude_remote_bash.diagnose.vectors.socket_matrix import check_socket_matrix

__all__ = [
    'run_diagnose',
    'vector_names',
]


# Ordered registry. Lookup by --vector name; iteration order is the run order.
_VECTORS: Mapping[str, Callable[[], VectorResult]] = {
    'config': check_config,
    'interpreter': check_interpreter,
    'local-network': check_local_network,
    'socket-matrix': check_socket_matrix,
}


def vector_names() -> tuple[str, ...]:
    return tuple(_VECTORS.keys())


def run_diagnose(vector_name: str | None = None) -> DiagnoseReport:
    """Run every vector, or just the one named via ``vector_name``."""
    if vector_name is None:
        return DiagnoseReport(results=[check() for check in _VECTORS.values()])

    check = _VECTORS.get(vector_name)
    if check is None:
        valid = ', '.join(_VECTORS.keys())
        raise ValueError(f'Unknown vector {vector_name!r}. Valid: {valid}')
    return DiagnoseReport(results=[check()])
