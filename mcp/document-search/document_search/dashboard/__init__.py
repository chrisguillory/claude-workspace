"""Dashboard entry point."""

from __future__ import annotations

import asyncio
import logging

from document_search.dashboard.server import DashboardServer
from document_search.dashboard.state import DashboardStateManager

__all__ = [
    'main',
]


def main() -> None:
    """Entry point for document-search-dashboard command."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    state_manager = DashboardStateManager()
    server = DashboardServer(state_manager)
    asyncio.run(server.run())
