"""Demo MCP server for dynamic tool registration at runtime.

Demonstrates MCP notifications/tools/list_changed by updating tool
descriptions every 60 seconds with a new timestamp and random color.
Claude Code picks up the changes between turns without /mcp reconnect.

Install:
    uv tool install --editable mcp/dynamic-tools-demo
    claude mcp add --scope user dynamic-tools-demo -- mcp-dynamic-tools-demo-server
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import sys
import typing
from datetime import datetime
from pathlib import Path

from fastmcp import Context, FastMCP
from fastmcp.tools import Tool
from mcp.server.session import ServerSession

__all__ = [
    'main',
]

logger = logging.getLogger(__name__)

LOG_DIR = Path.home() / '.claude-workspace' / 'dynamic-tools-demo'


def main() -> None:
    """Entry point for the Dynamic Tools Demo MCP server."""
    mcp.run()


class _SharedState:
    """Mutable state shared between tools and background task."""

    session: ServerSession | None = None


def _local_time() -> str:
    """Current local time with timezone label."""
    return datetime.now().astimezone().strftime('%H:%M:%S %Z')


def _log_event(event: str, **data: object) -> None:
    """Append a structured event to the log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {'timestamp': _local_time(), 'event': event, **data}
    with (LOG_DIR / 'events.jsonl').open('a') as f:
        f.write(json.dumps(entry) + '\n')
    logger.info('%s: %s', event, json.dumps(data) if data else '')


def _random_hex_color() -> str:
    """Generate a random hex color like #A3F29B."""
    return f'#{random.randint(0, 0xFFFFFF):06X}'


def _make_vibe_description() -> str:
    """Build a vibe_check description with current time and color."""
    now = _local_time()
    color = _random_hex_color()
    return f'Check the current vibe. Last updated: {now} — Color: {color}'


# -- Background Updater -------------------------------------------------------


async def _background_updater(mcp_instance: FastMCP, state: _SharedState) -> None:
    """Re-register vibe_check every 60s with updated description."""
    while True:
        await asyncio.sleep(60)
        if state.session is None:
            _log_event('updater_skipped', reason='no session captured yet')
            continue

        try:
            mcp_instance._local_provider.remove_tool('vibe_check')
        except KeyError:
            pass

        description = _make_vibe_description()

        async def vibe_check_fn() -> str:
            now = _local_time()
            color = _random_hex_color()
            return f'Vibe: {color} at {now}'

        tool = Tool.from_function(
            vibe_check_fn,
            name='vibe_check',
            description=description,
        )
        mcp_instance.add_tool(tool)

        await state.session.send_tool_list_changed()
        _log_event('updater_fired', description=description, notification='sent')


# -- Tool Registration --------------------------------------------------------


def _register_tools(mcp_instance: FastMCP, state: _SharedState) -> None:
    """Register static tools via closures over shared state."""

    initial_description = _make_vibe_description()
    _log_event('initial_registration', description=initial_description)

    @mcp_instance.tool(description=initial_description)
    async def vibe_check(ctx: Context) -> str:
        """Check the current vibe — returns time and a random color."""
        if state.session is None:
            state.session = ctx.session
            _log_event('session_captured')
        now = _local_time()
        color = _random_hex_color()
        return f'Vibe: {color} at {now}'

    @mcp_instance.tool()
    async def register_dynamic_tool(
        name: str,
        description: str,
        response_text: str,
        ctx: Context,
    ) -> str:
        """Create a new tool at runtime that returns a fixed response.

        Args:
            name: Tool name (lowercase, underscores)
            description: Tool description shown to the LLM
            response_text: Text the tool returns when called
        """
        if state.session is None:
            state.session = ctx.session
            _log_event('session_captured')

        async def dynamic_fn() -> str:
            return response_text

        tool = Tool.from_function(dynamic_fn, name=name, description=description)

        try:
            mcp_instance._local_provider.remove_tool(name)
        except KeyError:
            pass

        mcp_instance.add_tool(tool)
        await ctx.session.send_tool_list_changed()
        _log_event('tool_registered', name=name, description=description)
        return f"Registered tool '{name}'"

    @mcp_instance.tool()
    async def remove_dynamic_tool(name: str, ctx: Context) -> str:
        """Remove a previously registered dynamic tool.

        Args:
            name: Name of the tool to remove
        """
        if state.session is None:
            state.session = ctx.session
            _log_event('session_captured')

        try:
            mcp_instance._local_provider.remove_tool(name)
        except KeyError:
            return f"Tool '{name}' not found"

        await ctx.session.send_tool_list_changed()
        _log_event('tool_removed', name=name)
        return f"Removed tool '{name}'"


# -- Lifespan -----------------------------------------------------------------


@contextlib.asynccontextmanager
async def _lifespan(server_instance: FastMCP) -> typing.AsyncIterator[None]:
    """Server lifecycle: register tools, start background updater."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )

    # Clear log for fresh start
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / 'events.jsonl').write_text('')

    state = _SharedState()
    _register_tools(server_instance, state)

    task = asyncio.create_task(_background_updater(server_instance, state))
    _log_event('server_started')

    yield

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    _log_event('server_stopped')


mcp = FastMCP('dynamic-tools-demo', lifespan=_lifespan)
