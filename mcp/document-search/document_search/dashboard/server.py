"""Dashboard web server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from collections.abc import Mapping, Sequence

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from document_search.dashboard.state import DashboardStateManager
from document_search.paths import OPERATIONS_DIR
from document_search.schemas.dashboard import DashboardState, McpServer, OperationState

__all__ = [
    'DashboardServer',
]

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8765
MONITOR_INTERVAL_SECONDS = 5

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Search Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { margin-bottom: 30px; color: #1a1a1a; }
        h2 { font-size: 18px; color: #1a1a1a; margin-bottom: 16px; }
        .panel {
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        }
        .operation {
            border-bottom: 1px solid #eee;
            padding: 20px 0;
        }
        .operation:first-child { padding-top: 0; }
        .operation:last-child { border-bottom: none; padding-bottom: 0; }
        .op-header { margin-bottom: 12px; }
        .op-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .status-running { background: #e3f2fd; color: #1976d2; }
        .status-complete { background: #e8f5e9; color: #388e3c; }
        .status-failed { background: #ffebee; color: #d32f2f; }
        .op-meta { font-size: 13px; color: #666; }
        .progress-bar {
            background: #e8e8e8;
            border-radius: 6px;
            height: 28px;
            margin: 10px 0;
            position: relative;
            overflow: hidden;
        }
        .progress-fill {
            background: #34a853;
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 6px;
        }
        .progress-fill.embedded { background: #93d5a0; }
        .progress-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 13px;
            font-weight: 600;
            color: #1a1a1a;
            white-space: nowrap;
        }
        .pipeline {
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
            font-size: 13px;
            padding: 12px 16px;
            background: #f8f9fa;
            border-radius: 6px;
            margin: 12px 0;
            color: #444;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }
        .stat {
            padding: 12px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        .stat-label {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .stat-value {
            font-size: 20px;
            font-weight: 600;
            color: #1a1a1a;
        }
        .empty { text-align: center; padding: 40px; color: #999; }
        .server { padding: 8px 0; }
        .error-msg {
            color: #c62828;
            background: #ffebee;
            padding: 8px 12px;
            border-radius: 4px;
            margin-top: 8px;
            font-size: 13px;
            white-space: pre-wrap;
            font-family: monospace;
            max-height: 200px;
            overflow-y: auto;
        }
        .log-viewer {
            background: #1e1e1e;
            color: #d4d4d4;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 11px;
            line-height: 1.5;
            padding: 8px 12px;
            border-radius: 4px;
            margin-top: 8px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre;
        }
        .log-viewer .log-warn { color: #e2c08d; }
        .log-viewer .log-error { color: #f48771; }
        .log-viewer .log-info { color: #9cdcfe; }
        .log-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 8px;
            font-size: 12px;
            color: #666;
            cursor: pointer;
            user-select: none;
        }
        .log-header:hover { color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Document Search Dashboard</h1>

        <div class="panel">
            <h2>Active Operations</h2>
            <div id="active-ops"></div>
        </div>

        <div class="panel">
            <h2>Connected MCP Servers</h2>
            <div id="servers"></div>
        </div>

        <div class="panel">
            <h2>Recent Operations</h2>
            <div id="recent-ops"></div>
        </div>
    </div>

    <script>
        function formatOperation(op) {
            const p = op.progress;
            const status = op.ended_at ? (op.error ? 'failed' : 'complete') : 'running';

            const eta = p ? estimateEta(p) : null;
            const etaText = eta ? ` | ETA ${formatDuration(eta)}` : '';

            let progressHtml = '';
            if (p) {
                const filesPct = p.files_to_process > 0
                    ? (p.files_done / p.files_to_process * 100).toFixed(1)
                    : 0;
                const chunksEmbedPct = p.chunks_ingested > 0
                    ? (p.chunks_embedded / p.chunks_ingested * 100).toFixed(1)
                    : 0;
                const chunksStorePct = p.chunks_ingested > 0
                    ? (p.chunks_stored / p.chunks_ingested * 100).toFixed(1)
                    : 0;

                progressHtml = `
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${filesPct}%"></div>
                        <div class="progress-text">Files: ${p.files_done.toLocaleString()} / ${p.files_to_process.toLocaleString()} (${filesPct}%)</div>
                    </div>

                    <div class="progress-bar">
                        <div class="progress-fill embedded" style="width: ${chunksEmbedPct}%"></div>
                        <div class="progress-text">Chunks Embedded: ${p.chunks_embedded.toLocaleString()} / ${p.chunks_ingested.toLocaleString()}</div>
                    </div>

                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${chunksStorePct}%"></div>
                        <div class="progress-text">Chunks Stored: ${p.chunks_stored.toLocaleString()} / ${p.chunks_ingested.toLocaleString()}</div>
                    </div>

                    <div class="pipeline">
                        Pipeline: ${p.files_awaiting_chunk} → chunk → ${p.files_awaiting_embed} → embed → ${p.files_awaiting_store} → store
                    </div>

                    <div class="stats">
                        <div class="stat">
                            <div class="stat-label">Found</div>
                            <div class="stat-value">${p.files_found.toLocaleString()}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Cached</div>
                            <div class="stat-value">${p.files_cached.toLocaleString()}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">Errored</div>
                            <div class="stat-value">${p.files_errored}</div>
                        </div>
                        <div class="stat">
                            <div class="stat-label">429 Errors</div>
                            <div class="stat-value">${p.errors_429}</div>
                        </div>
                    </div>
                `;
            }

            const elapsed = op.result?.elapsed_seconds ?? p?.elapsed_seconds ?? 0;

            return `
                <div class="operation">
                    <div class="op-header">
                        <div class="op-title">
                            <span class="status-badge status-${status}">${status}</span>
                            <span>${op.collection_name}</span>
                        </div>
                        <div class="op-meta">
                            ${formatDuration(elapsed)} elapsed${etaText}
                            ${p ? `| Scan: ${p.scan_complete ? 'complete' : 'in progress'}` : ''}
                        </div>
                    </div>
                    <div class="op-meta">${op.directory}</div>
                    ${progressHtml}
                    ${op.error ? `<div class="error-msg">${op.error}</div>` : ''}
                    ${op.result ? `<div class="op-meta" style="margin-top: 8px;">Result: ${op.result.files_indexed} files, ${op.result.chunks_created} chunks</div>` : ''}
                    <div class="log-header" onclick="toggleLogs('${op.operation_id}')">
                        <span id="log-toggle-${op.operation_id}">▶ Logs</span>
                        <span id="log-count-${op.operation_id}"></span>
                    </div>
                    <div id="logs-${op.operation_id}" style="display:none"></div>
                </div>
            `;
        }

        function estimateEta(p) {
            if (p.status !== 'running' || p.chunks_stored === 0) return null;
            const rate = p.chunks_stored / p.elapsed_seconds;
            const remaining = p.chunks_ingested - p.chunks_stored;
            if (rate <= 0 || remaining <= 0) return null;
            return remaining / rate;
        }

        function formatDuration(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            if (h > 0) {
                return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
            }
            return `${m}:${s.toString().padStart(2, '0')}`;
        }

        const openLogs = new Set();

        function toggleLogs(opId) {
            const el = document.getElementById('logs-' + opId);
            const toggle = document.getElementById('log-toggle-' + opId);
            if (openLogs.has(opId)) {
                openLogs.delete(opId);
                el.style.display = 'none';
                toggle.textContent = '▶ Logs';
            } else {
                openLogs.add(opId);
                el.style.display = 'block';
                toggle.textContent = '▼ Logs';
                fetchLogs(opId);
            }
        }

        async function fetchLogs(opId) {
            try {
                const resp = await fetch(`/api/operations/${opId}/logs?tail=100`);
                const data = await resp.json();
                const el = document.getElementById('logs-' + opId);
                const countEl = document.getElementById('log-count-' + opId);
                if (!el) return;
                countEl.textContent = `${data.total_lines} lines`;
                if (data.lines.length === 0) {
                    el.innerHTML = '<div class="log-viewer">No logs yet</div>';
                    return;
                }
                const html = data.lines.map(line => {
                    let cls = '';
                    if (line.includes('[WARNING]')) cls = 'log-warn';
                    else if (line.includes('[ERROR]')) cls = 'log-error';
                    else if (line.includes('[INFO]')) cls = 'log-info';
                    const escaped = line.replace(/&/g,'&amp;').replace(/</g,'&lt;');
                    return cls ? `<span class="${cls}">${escaped}</span>` : escaped;
                }).join('\n');
                el.innerHTML = `<div class="log-viewer">${html}</div>`;
                // Auto-scroll to bottom
                el.querySelector('.log-viewer').scrollTop = el.querySelector('.log-viewer').scrollHeight;
            } catch (e) {
                console.error('Log fetch failed:', e);
            }
        }

        async function update() {
            try {
                const [activeResp, serversResp, recentResp] = await Promise.all([
                    fetch('/api/operations/active'),
                    fetch('/api/mcp-servers'),
                    fetch('/api/operations?limit=10')
                ]);

                const activeOps = await activeResp.json();
                const servers = await serversResp.json();
                const recentOps = await recentResp.json();

                // Active operations
                document.getElementById('active-ops').innerHTML =
                    activeOps.length === 0
                        ? '<div class="empty">No active operations</div>'
                        : activeOps.map(formatOperation).join('');

                // Servers
                document.getElementById('servers').innerHTML =
                    servers.length === 0
                        ? '<div class="empty">No MCP servers connected</div>'
                        : servers.map(s =>
                            `<div class="server">PID: <strong>${s.pid}</strong> | Started: ${new Date(s.started_at).toLocaleString()}</div>`
                        ).join('');

                // Recent completed
                const completedOps = recentOps.filter(o => o.ended_at);
                document.getElementById('recent-ops').innerHTML =
                    completedOps.length === 0
                        ? '<div class="empty">No recent operations</div>'
                        : completedOps.slice(0, 5).map(formatOperation).join('');
                // Refresh open log viewers
                for (const opId of openLogs) {
                    fetchLogs(opId);
                }
            } catch (e) {
                console.error('Update failed:', e);
            }
        }

        update();
        setInterval(update, 500);
    </script>
</body>
</html>"""


class DashboardServer:
    """Dashboard server with MCP server monitoring.

    MCP servers register themselves via DashboardStateManager.register_mcp_server().
    Dashboard monitors registered PIDs and exits when none are alive.
    """

    def __init__(self, state_manager: DashboardStateManager) -> None:
        self._state_manager = state_manager
        self._should_exit = False

    async def run(self) -> None:
        """Run server until all MCP servers exit."""
        previous_state = self._state_manager.load()
        existing_servers = previous_state.mcp_servers if previous_state else []
        preferred_port = previous_state.port if previous_state else None

        port = _find_port(preferred_port)

        state = DashboardState(
            port=port,
            server_pid=os.getpid(),
            mcp_servers=existing_servers,
        )
        self._state_manager.save(state)
        logger.info(f'Dashboard starting on http://127.0.0.1:{port}')

        monitor_task = asyncio.create_task(self._monitor_mcp_servers())

        config = uvicorn.Config(
            app=self._create_app(),
            host='127.0.0.1',
            port=port,
            log_level='warning',
        )
        server = uvicorn.Server(config)

        try:
            await server.serve()
        finally:
            monitor_task.cancel()
            self._state_manager.delete()

    async def _monitor_mcp_servers(self) -> None:
        """Shutdown when no MCP servers alive."""
        # Grace period for initial registration
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS * 2)

        while True:
            await asyncio.sleep(MONITOR_INTERVAL_SECONDS)
            live = self._state_manager.get_live_mcp_servers()

            if not live:
                logger.info('No live MCP servers, shutting down')
                # Graceful exit - raises SystemExit which uvicorn handles
                raise SystemExit(0)

    def _create_app(self) -> FastAPI:
        """Create FastAPI app."""
        app = FastAPI(title='Document Search Dashboard')
        state_manager = self._state_manager

        @app.get('/api/state')
        def get_state() -> DashboardState:
            state = state_manager.load()
            if state is None:
                raise HTTPException(status_code=503, detail='Dashboard state not available')
            return state

        @app.get('/api/mcp-servers')
        def get_mcp_servers() -> Sequence[McpServer]:
            return state_manager.get_live_mcp_servers()

        @app.get('/api/operations')
        def list_operations(limit: int = 20) -> Sequence[OperationState]:
            """List all operations (most recent first)."""
            ops = _read_operations()
            return ops[:limit]

        @app.get('/api/operations/active')
        def get_active_operations() -> Sequence[OperationState]:
            """Get currently running operations."""
            ops = _read_operations()
            return [o for o in ops if o.ended_at is None]

        @app.get('/api/operations/{operation_id}')
        def get_operation(operation_id: str) -> OperationState:
            """Get specific operation by ID."""
            file_path = OPERATIONS_DIR / f'{operation_id}.json'
            if not file_path.exists():
                raise HTTPException(status_code=404, detail='Operation not found')
            data = json.loads(file_path.read_text())
            return OperationState.model_validate(data)

        @app.get('/api/operations/{operation_id}/logs')
        def get_operation_logs(operation_id: str, tail: int = 50) -> Mapping[str, object]:
            """Get recent log lines for an operation.

            Args:
                operation_id: Operation UUID.
                tail: Number of lines from end of log to return.
            """
            log_path = OPERATIONS_DIR / f'{operation_id}.log'
            if not log_path.exists():
                return {'lines': [], 'total_lines': 0}
            text = log_path.read_text()
            all_lines = text.splitlines()
            return {
                'lines': all_lines[-tail:],
                'total_lines': len(all_lines),
            }

        @app.get('/', response_class=HTMLResponse)
        def index() -> str:
            return INDEX_HTML

        return app


def _find_port(preferred: int | None) -> int:
    """Find available port. Tries preferred, then default, then OS-assigned."""
    candidates = []
    if preferred:
        candidates.append(preferred)
    if preferred != DEFAULT_PORT:
        candidates.append(DEFAULT_PORT)

    for port in candidates:
        if _port_available(port):
            return port

    return _get_free_port()


def _port_available(port: int) -> bool:
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def _get_free_port() -> int:
    """Get OS-assigned free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port: int = s.getsockname()[1]
        return port


def _read_operations() -> Sequence[OperationState]:
    """Read all operation files from disk, sorted by created_at descending."""
    if not OPERATIONS_DIR.exists():
        return []

    ops: list[OperationState] = []
    for file_path in OPERATIONS_DIR.glob('*.json'):
        data = json.loads(file_path.read_text())
        ops.append(OperationState.model_validate(data))

    return sorted(ops, key=lambda o: o.created_at, reverse=True)
