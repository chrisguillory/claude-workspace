"""Dashboard web server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
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
        .timing-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            margin-top: 8px;
        }
        .timing-table th, .timing-table td {
            padding: 6px 10px;
            text-align: right;
            border-bottom: 1px solid #eee;
        }
        .timing-table th { text-align: left; color: #666; font-weight: 600; font-size: 11px; text-transform: uppercase; }
        .timing-table td:first-child { text-align: left; font-weight: 600; }
        .timing-table .sub-stage td:first-child { padding-left: 20px; color: #888; }
        .queue-chart { margin-top: 12px; }
        .queue-chart svg { width: 100%; height: 80px; }
        .chart-legend { display: flex; gap: 16px; font-size: 11px; color: #666; margin-top: 4px; }
        .chart-legend span { display: flex; align-items: center; gap: 4px; }
        .chart-legend .dot { width: 8px; height: 8px; border-radius: 50%; }
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
        .section-header { display: flex; justify-content: space-between; align-items: center; }
        .btn-clear {
            background: none;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 12px;
            color: #666;
            cursor: pointer;
        }
        .btn-clear:hover:not(:disabled) { background: #f5f5f5; color: #333; border-color: #ccc; }
        .btn-clear:disabled { opacity: 0.4; cursor: default; }
        .btn-delete {
            background: none;
            border: none;
            color: #999;
            cursor: pointer;
            font-size: 16px;
            padding: 0 4px;
            line-height: 1;
        }
        .btn-delete:hover { color: #d32f2f; }
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
            <div class="section-header">
                <h2>Connected MCP Servers</h2>
                <button class="btn-clear" onclick="restartDashboard()">Restart Dashboard</button>
            </div>
            <div id="servers"></div>
        </div>

        <div class="panel">
            <div class="section-header">
                <h2>Recent Operations</h2>
                <button id="btn-clear-all" class="btn-clear" onclick="clearAllOps()" disabled>Clear All</button>
            </div>
            <div id="recent-ops"></div>
        </div>
    </div>

    <script>
        function escapeHtml(s) {
            if (s == null) return '';
            return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                             .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
        }

        function formatOperationMeta(op) {
            const p = op.progress;
            const status = op.ended_at ? (op.error ? 'failed' : 'complete') : 'running';
            const elapsed = status === 'running'
                ? (Date.now() - new Date(op.created_at).getTime()) / 1000
                : (op.result?.elapsed_seconds ?? p?.elapsed_seconds ?? 0);
            const eta = p ? estimateEta(p) : null;
            const etaText = eta ? ` | ETA ${formatDuration(eta)}` : '';
            return `${formatDuration(elapsed)} elapsed${etaText}${p ? ` | Scan: ${p.scan_complete ? 'complete' : 'in progress'}` : ''}`;
        }

        function formatOperationProgressHtml(op) {
            const p = op.progress;
            if (!p) return '';
            if (!p.scan_complete) {
                const cachedText = p.files_cached > 0 ? `, ${p.files_cached.toLocaleString()} cached` : '';
                return `<div class="pipeline">Scanning: ${p.files_found.toLocaleString()} files found${cachedText}...</div>`;
            }
            const filesPct = p.files_to_process > 0
                ? (p.files_done / p.files_to_process * 100).toFixed(1) : 0;
            const chunksEmbedPct = p.chunks_ingested > 0
                ? (p.chunks_embedded / p.chunks_ingested * 100).toFixed(1) : 0;
            const chunksStorePct = p.chunks_ingested > 0
                ? (p.chunks_stored / p.chunks_ingested * 100).toFixed(1) : 0;
            const totalChunked = p.chunks_ingested + (p.chunks_skipped || 0);
            const chunkSkipPct = totalChunked > 0
                ? ((p.chunks_skipped || 0) / totalChunked * 100).toFixed(1) : 0;
            const chunkCacheText = (p.chunks_skipped || 0) > 0
                ? `<div class="pipeline" style="color:#2e7d32;background:#e8f5e9">Chunk cache: ${(p.chunks_skipped).toLocaleString()} of ${totalChunked.toLocaleString()} chunks unchanged (${chunkSkipPct}% reused)</div>`
                : '';
            return `
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
                ${chunkCacheText}
                <div class="pipeline">
                    Pipeline: ${p.files_awaiting_chunk} queued → chunk${p.files_in_chunk ? ` (${p.files_in_chunk} active)` : ''} → ${p.files_awaiting_embed} queued → embed${p.files_in_embed ? ` (${p.files_in_embed} active)` : ''} → ${p.files_awaiting_store} queued → store${p.files_in_store ? ` (${p.files_in_store} active)` : ''}
                </div>
                <div class="stats">
                    <div class="stat"><div class="stat-label">Found</div><div class="stat-value">${p.files_found.toLocaleString()}</div></div>
                    <div class="stat"><div class="stat-label">Cached</div><div class="stat-value">${p.files_cached.toLocaleString()}</div></div>
                    <div class="stat"><div class="stat-label">Chunks Skipped</div><div class="stat-value">${(p.chunks_skipped || 0).toLocaleString()}</div></div>
                    <div class="stat"><div class="stat-label">Cache Hits</div><div class="stat-value">${p.embed_cache_hits.toLocaleString()}</div></div>
                    <div class="stat"><div class="stat-label">Cache Misses</div><div class="stat-value">${p.embed_cache_misses.toLocaleString()}</div></div>
                    <div class="stat"><div class="stat-label">Errored</div><div class="stat-value">${p.files_errored}</div></div>
                    <div class="stat"><div class="stat-label">429 Errors</div><div class="stat-value">${p.errors_429}</div></div>
                </div>
                ${p.by_file_type && Object.keys(p.by_file_type).length > 0 ? formatFileTypeChart(p.by_file_type) : ''}
                ${formatQueueDepthChart(p.queue_depth_series)}
            `;
        }

        function formatCompletedSummary(op) {
            const r = op.result;
            if (!r) return '';
            const rate = r.elapsed_seconds > 0 ? (r.files_indexed / r.elapsed_seconds).toFixed(1) : '0';
            const totalChunks = r.chunks_created + (r.chunks_skipped || 0);
            const chunksText = (r.chunks_skipped || 0) > 0
                ? `${totalChunks.toLocaleString()} chunks (${(r.chunks_skipped).toLocaleString()} reused)`
                : `${r.chunks_created.toLocaleString()} chunks`;
            const parts = [
                `${formatDuration(r.elapsed_seconds)}`,
                `${r.files_scanned.toLocaleString()} files \\u2192 ${chunksText}`,
                `${rate}/s`,
            ];
            let line2Parts = [`Cache: ${r.embed_cache_hits.toLocaleString()} hits, ${r.embed_cache_misses.toLocaleString()} misses`];
            if ((r.chunks_skipped || 0) > 0) {
                const skipPct = (r.chunks_skipped / totalChunks * 100).toFixed(0);
                line2Parts.push(`Chunk cache: ${skipPct}%`);
            }
            const errCount = r.errors ? r.errors.length : 0;
            if (errCount > 0) line2Parts.push(`<span style="color:#c62828">Errors: ${errCount}</span>`);
            if (r.stopped_after) line2Parts.push(`Stopped after ${r.stopped_after}`);
            return `
                <div style="font-size:12px;color:#666;margin-top:4px;line-height:1.6">
                    <div>${parts.join(' | ')}</div>
                    <div>${line2Parts.join(' | ')}</div>
                </div>`;
        }

        function formatOperation(op) {
            const status = op.ended_at ? (op.error ? 'failed' : 'complete') : 'running';
            const isComplete = op.ended_at && !op.error;
            return `
                <div class="operation">
                    <div class="op-header">
                        <div class="op-title">
                            <span class="status-badge status-${status}">${status}</span>
                            <span>${escapeHtml(op.collection_name)}</span>
                            ${status !== 'running' ? `<button class="btn-delete" onclick="deleteOp('${op.operation_id}')" title="Delete">×</button>` : ''}
                        </div>
                        ${!isComplete ? `<div class="op-meta" id="meta-${op.operation_id}">${formatOperationMeta(op)}</div>` : ''}
                    </div>
                    <div class="op-meta">${escapeHtml(op.directory)}</div>
                    ${isComplete ? formatCompletedSummary(op) : `<div id="progress-${op.operation_id}">${formatOperationProgressHtml(op)}</div>`}
                    ${op.error ? `<div class="error-msg">${escapeHtml(op.error)}</div>` : ''}
                    ${op.result?.by_file_type && Object.keys(op.result.by_file_type).length > 0 ? formatFileTypeChart(op.result.by_file_type) : ''}
                    ${op.result?.timing ? formatTimingReport(op.result.timing, op.operation_id) : ''}
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

        function formatFileTypeSummary(summary) {
            // "scanned=44655 cached=3" → "44,655 scanned · 3 cached"
            return summary.split(' ').map(pair => {
                const [key, val] = pair.split('=');
                return parseInt(val).toLocaleString() + ' ' + key;
            }).join(' · ');
        }

        // File type to extensions mapping
        const FILE_TYPE_EXTS = {
            'image': '.jpg .jpeg .png .heic .gif',
            'json': '.json',
            'text': '.txt',
            'markdown': '.md',
            'jsonl': '.jsonl .ndjson',
            'csv': '.csv',
            'pdf': '.pdf',
            'email': '.eml',
        };

        // Earth palette [light, dark] per type
        const EARTH_PALETTE = [
            ['#d4a373','#8b5e34'], ['#6b9080','#3a5a40'], ['#c1666b','#8b3a3a'],
            ['#e8ac65','#b5651d'], ['#7ea8be','#4a7c94'], ['#a98467','#6f4e37'],
            ['#d4a5a5','#9e6b6b'], ['#8fbc8f','#556b2f'],
        ];

        function formatFileTypeChart(byFileType) {
            // Parse "scanned=X cached=Y" into structured data
            const items = Object.entries(byFileType).map(([ft, summary]) => {
                const parts = {};
                summary.split(' ').forEach(pair => {
                    const [k, v] = pair.split('=');
                    parts[k] = parseInt(v);
                });
                return { type: ft, scanned: parts.scanned || 0, cached: parts.cached || 0 };
            }).sort((a, b) => b.scanned - a.scanned);

            const maxScanned = Math.max(...items.map(d => d.scanned));
            const total = items.reduce((s, d) => s + d.scanned, 0);
            const BAR_MAX_H = 130;

            return `
                <div style="display:flex;align-items:flex-end;gap:2px;border-bottom:2px solid #e8e8e8;margin-top:12px">
                    ${items.map((d, i) => {
                        const bh = Math.max(2, d.scanned / maxScanned * BAR_MAX_H);
                        const ch = d.cached > 0 ? Math.max(3, d.cached / d.scanned * bh) : 0;
                        const pct = (d.scanned / total * 100).toFixed(0);
                        const [lt, dk] = EARTH_PALETTE[i % EARTH_PALETTE.length];
                        return `<div style="flex:1;display:flex;flex-direction:column;align-items:center">
                            <div style="font-size:10px;color:#999;margin-bottom:1px">${pct}%</div>
                            <div style="font-size:11px;font-weight:600;color:#444;margin-bottom:2px;font-variant-numeric:tabular-nums">${d.scanned.toLocaleString()}</div>
                            <div style="width:100%;height:${bh}px;border-radius:4px 4px 0 0;position:relative;overflow:hidden;background:linear-gradient(to bottom,${lt},${dk}44)">
                                ${ch > 0 ? `<div style="position:absolute;bottom:0;width:100%;height:${ch}px;background:${dk};opacity:0.7"></div>` : ''}
                            </div>
                        </div>`;
                    }).join('')}
                </div>
                <div style="display:flex;gap:2px;margin-top:6px">
                    ${items.map((d, i) => {
                        const [, dk] = EARTH_PALETTE[i % EARTH_PALETTE.length];
                        const exts = FILE_TYPE_EXTS[d.type] || '';
                        return `<div style="flex:1;text-align:center">
                            <div style="font-size:11px;font-weight:600;text-transform:uppercase;color:#555">${d.type}</div>
                            <div style="font-size:9px;color:#bbb">${exts}</div>
                            <div style="font-size:9px;font-weight:600;color:${d.cached > 0 ? dk : '#ccc'};margin-top:1px">${d.cached.toLocaleString()} cached</div>
                        </div>`;
                    }).join('')}
                </div>
            `;
        }

        function formatQueueDepthChart(series) {
            if (!series || series.length < 2) return '';
            const maxT = series[series.length - 1].elapsed_seconds;
            const maxQ = Math.max(1, ...series.map(s => Math.max(s.file_queue, s.embed_queue, s.upsert_queue)));
            const ML = 44, MR = 8, MT = 4, MB = 18;  // margins for axes (ML=44 matches interactive charts)
            const W = 600, H = 90;
            const pw = W - ML - MR, ph = H - MT - MB;  // plot area
            const x = (t) => ML + (t / maxT) * pw;
            const y = (v) => MT + ph - (v / maxQ) * ph;
            const line = (data, key) => data.map((s, i) => `${i===0?'M':'L'}${x(s.elapsed_seconds).toFixed(1)},${y(s[key]).toFixed(1)}`).join(' ');

            // Y-axis grid lines and labels
            let yTicks = '';
            const yTickCount = Math.min(4, Math.max(1, Math.floor(maxQ / 50)));
            for (let i = 1; i <= yTickCount; i++) {
                const val = Math.round(maxQ * i / (yTickCount + 1));
                const yp = y(val);
                yTicks += `<line x1="${ML}" y1="${yp}" x2="${ML + pw}" y2="${yp}" stroke="#e0e0e0" stroke-width="0.5" stroke-dasharray="2"/>`;
                yTicks += `<text x="${ML - 4}" y="${yp + 3}" text-anchor="end" font-size="8" fill="#999">${val.toLocaleString()}</text>`;
            }

            // X-axis time ticks (up to 5)
            const tickCount = Math.min(5, Math.max(1, Math.floor(maxT / 10)));
            const tickInterval = tickCount > 0 ? maxT / tickCount : maxT;
            let xTicks = '';
            for (let i = 0; i <= tickCount; i++) {
                const t = i * tickInterval;
                const tx = x(t);
                const label = t < 60 ? `${Math.round(t)}s` : `${(t/60).toFixed(1)}m`;
                xTicks += `<line x1="${tx}" y1="${MT+ph}" x2="${tx}" y2="${MT+ph+3}" stroke="#ccc" stroke-width="0.5"/>`;
                xTicks += `<text x="${tx}" y="${H-2}" text-anchor="middle" font-size="8" fill="#999">${label}</text>`;
            }

            return `
                <div class="queue-chart">
                    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">
                        <text x="${ML-4}" y="${MT+4}" text-anchor="end" font-size="8" fill="#999">${maxQ.toLocaleString()}</text>
                        <text x="${ML-4}" y="${MT+ph}" text-anchor="end" font-size="8" fill="#999">0</text>
                        ${yTicks}
                        <line x1="${ML}" y1="${MT+ph}" x2="${ML+pw}" y2="${MT+ph}" stroke="#e8e8e8" stroke-width="0.5"/>
                        <line x1="${ML}" y1="${MT}" x2="${ML+pw}" y2="${MT}" stroke="#e8e8e8" stroke-width="0.5" stroke-dasharray="2"/>
                        ${xTicks}
                        <path d="${line(series, 'file_queue')}" fill="none" stroke="#e65100" stroke-width="1" opacity="0.7"/>
                        <path d="${line(series, 'embed_queue')}" fill="none" stroke="#1976d2" stroke-width="1" opacity="0.7"/>
                        <path d="${line(series, 'upsert_queue')}" fill="none" stroke="#388e3c" stroke-width="1" opacity="0.7"/>
                    </svg>
                    <div class="chart-legend">
                        <span><span class="dot" style="background:#e65100"></span>chunk (peak ${Math.max(...series.map(s=>s.file_queue)).toLocaleString()} files)</span>
                        <span><span class="dot" style="background:#1976d2"></span>embed (peak ${Math.max(...series.map(s=>s.embed_queue)).toLocaleString()} files)</span>
                        <span><span class="dot" style="background:#388e3c"></span>store (peak ${Math.max(...series.map(s=>s.upsert_queue)).toLocaleString()} files)</span>
                    </div>
                </div>`;
        }

        function createTimeSeriesChart(containerId, data, queueData, totalElapsed, aggStages) {
            const el = document.getElementById(containerId);
            if (!el || !data || data.length === 0) return;

            const HEIGHT_PRESETS = {S: {ph: 120, qh: 70}, M: {ph: 158, qh: 90}, L: {ph: 240, qh: 130}, XL: {ph: 340, qh: 180}, XXL: {ph: 480, qh: 240}};
            const state = {
                start: 0, end: totalElapsed,
                vis: new Set(['chunk', 'embed', 'embed_sparse', 'embed_dense', 'store']),
                qvis: new Set(['file_queue', 'embed_queue', 'upsert_queue', 'chunk_inflight', 'embed_inflight', 'store_inflight', 'chunk_done', 'embed_done', 'store_done']),
                pct: 0.50, wins: {}, ws: 5, wsMode: 'auto', logScale: false, heightKey: 'M',
                y2v: null, qMaxY: 0, yBounds: null, px: null, py: null,
            };

            const THEME = {
                colors: {grid: '#e0e0e0', gridVert: '#e8e8e8', axis: '#ccc', text: '#999', textMuted: '#aaa', border: '#ddd'},
                spacing: {ml: 44, mr: 28, mt: 4, mb: 20, tickLen: 3, dataPad: 4},
                type: {axis: 8, axisSmall: 7, legend: 11},
                chart: {dotR: 1.0, dotHoverR: 3.5, lineW: 1, lineOp: 0.5, dotOp: 0.85, qLineW: 1, qLineOp: 0.7},
            };
            const COLORS = {embed_sparse:'#c62828', embed_dense:'#1565c0', store:'#2e7d32', embed:'#6a1b9a', chunk:'#78909c'};
            const QCOLORS = {file_queue:'#e65100', embed_queue:'#1976d2', upsert_queue:'#388e3c', chunk_inflight:'#ff6f00', embed_inflight:'#1e88e5', store_inflight:'#43a047', cumulative:'#6a1b9a', chunk_done:'#ef5350', embed_done:'#42a5f5', store_done:'#66bb6a'};
            const LABELS = {chunk:'chunk', embed:'embed', embed_sparse:'sparse', embed_dense:'dense', store:'store'};
            const W = 600, ML = THEME.spacing.ml, MR = THEME.spacing.mr;
            const PT = THEME.spacing.mt, PB = THEME.spacing.mb;
            const QT = THEME.spacing.mt, QB = THEME.spacing.mb;
            const pw = W - ML - MR;
            const PCT_LABELS = {0.50:'p50', 0.95:'p95', 0.99:'p99', 1.0:'max'};

            function lowerBound(arr, v) {
                let lo = 0, hi = arr.length;
                while (lo < hi) { const m = (lo + hi) >> 1; arr[m] < v ? lo = m + 1 : hi = m; }
                return lo;
            }

            function chooseWinSize(range) {
                for (const s of [300, 120, 60, 30, 15, 10, 5, 2, 1, 0.5, 0.25, 0.1]) {
                    const n = range / s; if (n >= 10 && n <= 30) return s;
                }
                return Math.max(0.1, Math.round(range / 20 * 10) / 10);
            }

            function percentile(sorted, p) {
                if (!sorted.length) return null;
                return sorted[Math.min(Math.floor(sorted.length * p), sorted.length - 1)];
            }

            function computeWindows(comp, dur, ws, r0, r1) {
                const wins = [];
                for (let s = r0; s < r1; s += ws) {
                    const e = Math.min(s + ws, r1);
                    const lo = lowerBound(comp, s), hi = lowerBound(comp, e);
                    if (hi > lo) {
                        const b = dur.slice(lo, hi).sort((a, c) => a - c);
                        wins.push({s, e, v: percentile(b, state.pct), n: hi - lo});
                    } else {
                        wins.push({s, e, v: null, n: 0});
                    }
                }
                return wins;
            }

            function dataBounds(allWins) {
                let mn = Infinity, mx = -Infinity;
                for (const ws of Object.values(allWins)) {
                    for (const w of ws) { if (w.v !== null) { mn = Math.min(mn, w.v); mx = Math.max(mx, w.v); } }
                }
                return {rawMn: mn, rawMx: mx};
            }

            function logBounds(rawMn, rawMx) {
                if (rawMn === Infinity) return {mn: 1, mx: 1000};
                // Extend half a decade below the data minimum so the lowest
                // power-of-10 gridline is visible above the plot floor.
                const l0 = Math.floor(Math.log10(rawMn)) - 0.5;
                const l1 = Math.ceil(Math.log10(rawMx));
                return {mn: Math.pow(10, l0), mx: Math.pow(10, l1 <= Math.ceil(l0) ? Math.ceil(l0) + 1 : l1)};
            }

            function linearBounds(rawMn, rawMx) {
                if (rawMn === Infinity) return {mn: 0, mx: 1000};
                // Nice-number rounding with 10% top padding for visual clearance
                const range = rawMx - Math.min(0, rawMn);
                if (range === 0) return {mn: 0, mx: rawMx * 1.1 || 1};
                const mag = Math.pow(10, Math.floor(Math.log10(range)));
                const norm = range / mag;
                const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
                const niceMax = nice * mag;
                return {mn: 0, mx: niceMax};
            }

            function fmtMs(ms) {
                if (ms >= 60000) return (ms / 60000).toFixed(1) + 'm';
                if (ms >= 1000) return (ms / 1000).toFixed(1) + 's';
                if (ms >= 1) return Math.round(ms) + 'ms';
                return ms.toFixed(2) + 'ms';
            }
            function fmtT(s) { return s >= 60 ? (s / 60).toFixed(1) + 'm' : s.toFixed(1) + 's'; }

            function aggP50(stageId) {
                const s = aggStages.find(x => x.stage === stageId);
                return s ? fmtMs(s.processing.p50_ms) : '';
            }

            function queueAt(t) {
                if (!queueData || !queueData.length) return null;
                const qs = queueData;
                if (t < qs[0].elapsed_seconds) return null;
                if (t <= qs[0].elapsed_seconds) return qs[0];
                if (t >= qs[qs.length - 1].elapsed_seconds) return qs[qs.length - 1];
                let lo = 0, hi = qs.length - 1;
                while (lo < hi - 1) { const m = (lo + hi) >> 1; qs[m].elapsed_seconds <= t ? lo = m : hi = m; }
                const a = qs[lo], b = qs[hi], f = (t - a.elapsed_seconds) / ((b.elapsed_seconds - a.elapsed_seconds) || 1);
                return {
                    file_queue: Math.round(a.file_queue + f * (b.file_queue - a.file_queue)),
                    embed_queue: Math.round(a.embed_queue + f * (b.embed_queue - a.embed_queue)),
                    upsert_queue: Math.round(a.upsert_queue + f * (b.upsert_queue - a.upsert_queue)),
                    chunk_in_flight: Math.round(a.chunk_in_flight + f * (b.chunk_in_flight - a.chunk_in_flight)),
                    embed_in_flight: Math.round(a.embed_in_flight + f * (b.embed_in_flight - a.embed_in_flight)),
                    store_in_flight: Math.round(a.store_in_flight + f * (b.store_in_flight - a.store_in_flight)),
                    files_chunk_done: Math.round(a.files_chunk_done + f * (b.files_chunk_done - a.files_chunk_done)),
                    files_embed_done: Math.round(a.files_embed_done + f * (b.files_embed_done - a.files_embed_done)),
                    files_store_done: Math.round(a.files_store_done + f * (b.files_store_done - a.files_store_done)),
                };
            }

            function t2x(t) { return ML + ((t - state.start) / (state.end - state.start || 1)) * pw; }
            function x2t(x) { return state.start + ((x - ML) / pw) * (state.end - state.start); }

            function generateXAxis(r0, r1, left, right, bottom, labelY, fmt, showGrid) {
                const range = r1 - r0;
                const rawTc = Math.min(6, Math.max(2, Math.floor(range / 5)));
                const rawInterval = range / rawTc;
                const mag = Math.pow(10, Math.floor(Math.log10(rawInterval)));
                const norm = rawInterval / mag;
                const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
                const interval = nice * mag;
                const tc = Math.ceil(range / interval);
                const px = (t) => left + ((t - r0) / range) * (right - left);
                let svg = '';
                for (let i = 0; i <= tc; i++) {
                    const t = Math.min(r0 + i * interval, r1);
                    if (t > r1) break;
                    const x = px(t);
                    svg += '<line x1="' + x + '" y1="' + (bottom - THEME.spacing.tickLen) + '" x2="' + x + '" y2="' + (bottom + THEME.spacing.tickLen) + '" stroke="' + THEME.colors.axis + '" stroke-width="0.5"/>';
                    if (showGrid && i > 0) svg += '<line x1="' + x + '" y1="' + (PT) + '" x2="' + x + '" y2="' + bottom + '" stroke="' + THEME.colors.gridVert + '" stroke-width="1" stroke-dasharray="3,2"/>';
                    svg += '<text x="' + x + '" y="' + labelY + '" text-anchor="middle" font-size="' + THEME.type.axis + '" fill="' + THEME.colors.text + '">' + fmt(t) + '</text>';
                }
                return {svg, px};
            }

            function generateYAxis(min, max, logScale, top, bottom, left, right, fmt, rawMin) {
                let py, svg = '';
                const dp = THEME.spacing.dataPad;  // Both modes get padding for clearance
                const h = bottom - top - dp, w = right - left;
                if (logScale) {
                    const lmin = Math.log10(min), lmax = Math.log10(max), lr = lmax - lmin || 1;
                    py = (v) => top + h - ((Math.log10(Math.max(min, v)) - lmin) / lr) * h;
                    for (let p = Math.ceil(lmin); p <= Math.floor(lmax); p++) {
                        const v = Math.pow(10, p), y = py(v);
                        svg += '<line x1="' + left + '" y1="' + y + '" x2="' + (left + w) + '" y2="' + y + '" stroke="' + THEME.colors.grid + '" stroke-width="1" stroke-dasharray="2"/>';
                        svg += '<text x="' + (left - 4) + '" y="' + (y + 3) + '" text-anchor="end" font-size="' + THEME.type.axis + '" fill="' + THEME.colors.text + '">' + fmt(v) + '</text>';
                    }
                    // Show actual data minimum with label inside the right margin
                    if (rawMin !== undefined && rawMin !== Infinity) {
                        const yMin = py(rawMin);
                        svg += '<line x1="' + left + '" y1="' + yMin + '" x2="' + (left + w) + '" y2="' + yMin + '" stroke="' + THEME.colors.border + '" stroke-width="0.5" stroke-dasharray="1,2"/>';
                        svg += '<text x="' + (left + w + 3) + '" y="' + (yMin + 3) + '" text-anchor="start" font-size="' + THEME.type.axisSmall + '" fill="' + THEME.colors.textMuted + '">' + fmt(rawMin) + '</text>';
                    }
                } else {
                    py = (v) => top + h - (v / (max || 1)) * h;
                    // Nice tick intervals for linear scale
                    const rawRange = max;
                    const rawTc = 5;
                    const rawInterval = rawRange / rawTc;
                    const mag = rawRange > 0 ? Math.pow(10, Math.floor(Math.log10(rawInterval || 1))) : 1;
                    const norm = rawInterval / mag;
                    const nice = norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10;
                    const interval = nice * mag;
                    const tc = Math.ceil(rawRange / interval);
                    for (let i = 0; i <= tc; i++) {
                        const v = i * interval, y = py(v);
                        if (v > max * 1.01) break;
                        svg += '<line x1="' + left + '" y1="' + y + '" x2="' + (left + w) + '" y2="' + y + '" stroke="' + THEME.colors.grid + '" stroke-width="1" stroke-dasharray="2"/>';
                        svg += '<text x="' + (left - 4) + '" y="' + (y + 3) + '" text-anchor="end" font-size="' + THEME.type.axis + '" fill="' + THEME.colors.text + '">' + fmt(v) + '</text>';
                    }
                }
                return {svg, py};
            }

            let raf = null;
            function scheduleRender() { if (!raf) raf = requestAnimationFrame(() => { raf = null; render(); }); }

            let dragStart = null;

            let tooltip = document.getElementById('tstip-' + containerId);
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.id = 'tstip-' + containerId;
                tooltip.style.cssText = 'display:none;position:fixed;background:rgba(30,30,30,0.95);color:#fff;padding:8px 12px;border-radius:6px;font-size:11px;font-family:monospace;pointer-events:none;z-index:1000;box-shadow:0 2px 8px rgba(0,0,0,0.3);line-height:1.6';
                document.body.appendChild(tooltip);
            }

            function render() {
                const range = state.end - state.start;
                state.ws = state.wsMode === 'auto' ? chooseWinSize(range) : parseFloat(state.wsMode);

                // Compute dimensions from height preset
                const hp = HEIGHT_PRESETS[state.heightKey];
                const PH = hp.ph, QH = hp.qh;
                const ph = PH - PT - PB, qph = QH - QT - QB;

                state.wins = {};
                for (const sd of data) {
                    if (!state.vis.has(sd.stage)) continue;
                    state.wins[sd.stage] = computeWindows(sd.completions, sd.durations, state.ws, state.start, state.end);
                }

                const db = dataBounds(state.wins);
                const yb = state.logScale ? logBounds(db.rawMn, db.rawMx) : linearBounds(db.rawMn, db.rawMx);
                state.yBounds = yb;

                // Store Y-inverse function for hover
                if (state.logScale) {
                    const lmin = Math.log10(yb.mn), lmax = Math.log10(yb.mx), lr = lmax - lmin || 1;
                    state.y2v = (yPix) => Math.pow(10, lmin + ((PT + ph - yPix) / ph) * lr);
                } else {
                    state.y2v = (yPix) => ((PT + ph - yPix) / ph) * yb.mx;
                }

                // Generate axes with nice-number rounding and gridlines
                const yAxis = generateYAxis(yb.mn, yb.mx, state.logScale, PT, PT + ph, ML, ML + pw, fmtMs, db.rawMn);
                const py = yAxis.py;
                state.py = py;
                const xAxis = generateXAxis(state.start, state.end, ML, ML + pw, PT + ph, PH - 2, fmtT, true);
                const px = xAxis.px;
                state.px = px;

                // Dot+line chart (dots at window midpoints, thin connecting lines)
                let paths = '';
                for (const [sid, wins] of Object.entries(state.wins)) {
                    const pts = [];
                    let lastEnd = -Infinity;
                    for (const w of wins) {
                        if (w.v === null) continue;
                        const cx = px((w.s + w.e) / 2).toFixed(1);
                        const cy = py(w.v).toFixed(1);
                        pts.push({cx, cy, brk: (w.s - lastEnd) > state.ws * 2});
                        lastEnd = w.e;
                    }
                    // Connecting lines (thin, behind dots, break on large gaps)
                    let ld = '';
                    for (const p of pts) {
                        ld += (p.brk ? (ld ? ' M' : 'M') : ' L') + p.cx + ',' + p.cy;
                    }
                    if (ld) paths += '<path d="' + ld + '" fill="none" stroke="' + COLORS[sid] + '" stroke-width="' + THEME.chart.lineW + '" opacity="' + THEME.chart.lineOp + '"/>';
                    // Dots (on top of lines)
                    for (const p of pts) {
                        paths += '<circle cx="' + p.cx + '" cy="' + p.cy + '" r="' + THEME.chart.dotR + '" fill="' + COLORS[sid] + '" opacity="' + THEME.chart.dotOp + '"/>';
                    }
                }

                // Queue depth chart
                const vqs = queueData.filter(s => s.elapsed_seconds >= state.start && s.elapsed_seconds <= state.end);
                let qSvg = '';
                if (vqs.length >= 2) {
                    const rawMaxQ = Math.max(1,
                        ...vqs.map(s => Math.max(
                            state.qvis.has('file_queue') ? s.file_queue : 0,
                            state.qvis.has('embed_queue') ? s.embed_queue : 0,
                            state.qvis.has('upsert_queue') ? s.upsert_queue : 0,
                            state.qvis.has('chunk_inflight') ? s.chunk_in_flight : 0,
                            state.qvis.has('embed_inflight') ? s.embed_in_flight : 0,
                            state.qvis.has('store_inflight') ? s.store_in_flight : 0,
                            state.qvis.has('chunk_done') ? (s.files_chunk_done || 0) : 0,
                            state.qvis.has('embed_done') ? (s.files_embed_done || 0) : 0,
                            state.qvis.has('store_done') ? (s.files_store_done || 0) : 0
                        ))
                    );
                    const maxQ = rawMaxQ * 1.05;
                    state.qMaxY = maxQ;
                    // Generate queue chart axes
                    const qYAxis = generateYAxis(0, maxQ, false, QT, QT + qph, ML, ML + pw, (v) => v.toLocaleString());
                    const qy = qYAxis.py;
                    const qXAxis = generateXAxis(state.start, state.end, ML, ML + pw, QT + qph, QH - 2, fmtT, true);
                    const qx = qXAxis.px;
                    const qLine = (d, k) => d.map((s, i) => (i === 0 ? 'M' : 'L') + qx(s.elapsed_seconds).toFixed(1) + ',' + qy(s[k]).toFixed(1)).join(' ');
                    let qPaths = '';
                    // Queue lines (solid)
                    const queueKeys = ['file_queue', 'embed_queue', 'upsert_queue'];
                    for (const k of queueKeys) {
                        if (state.qvis.has(k)) qPaths += '<path d="' + qLine(vqs, k) + '" fill="none" stroke="' + QCOLORS[k] + '" stroke-width="' + THEME.chart.qLineW + '" opacity="' + THEME.chart.qLineOp + '"/>';
                    }
                    // In-flight lines (dashed)
                    const inflightKeys = ['chunk_inflight', 'embed_inflight', 'store_inflight'];
                    for (const k of inflightKeys) {
                        if (state.qvis.has(k)) qPaths += '<path d="' + qLine(vqs, k.replace('_inflight', '_in_flight')) + '" fill="none" stroke="' + QCOLORS[k] + '" stroke-width="' + THEME.chart.qLineW + '" opacity="' + (THEME.chart.qLineOp * 0.85) + '" stroke-dasharray="3,2"/>';
                    }
                    // Completion curves (solid)
                    const doneMap = {chunk_done: 'files_chunk_done', embed_done: 'files_embed_done', store_done: 'files_store_done'};
                    for (const k of Object.keys(doneMap)) {
                        if (state.qvis.has(k)) qPaths += '<path d="' + qLine(vqs, doneMap[k]) + '" fill="none" stroke="' + QCOLORS[k] + '" stroke-width="' + THEME.chart.qLineW + '" opacity="' + THEME.chart.qLineOp + '"/>';
                    }
                    qSvg = '<svg viewBox="0 0 ' + W + ' ' + QH + '" preserveAspectRatio="xMidYMid meet" style="width:100%;cursor:crosshair" class="ts-qchart">' +
                        '<rect x="' + ML + '" y="' + QT + '" width="' + pw + '" height="' + qph + '" fill="#fafafa" stroke="' + THEME.colors.border + '" stroke-width="1"/>' +
                        qYAxis.svg +
                        '<line x1="' + ML + '" y1="' + (QT + qph) + '" x2="' + (ML + pw) + '" y2="' + (QT + qph) + '" stroke="' + THEME.colors.axis + '" stroke-width="1"/>' +
                        '<line x1="' + ML + '" y1="' + QT + '" x2="' + ML + '" y2="' + (QT + qph) + '" stroke="' + THEME.colors.axis + '" stroke-width="1"/>' +
                        qXAxis.svg + qPaths +
                        '<line class="ts-qcross" x1="0" y1="' + QT + '" x2="0" y2="' + (QT + qph) + '" stroke="#999" stroke-width="0.5" stroke-dasharray="2" display="none"/>' +
                        '<line class="ts-qhcross" x1="' + ML + '" y1="0" x2="' + (ML + pw) + '" y2="0" stroke="#999" stroke-width="0.5" stroke-dasharray="2" display="none"/>' +
                        '<rect class="ts-qylabel-bg" x="2" y="0" width="38" height="14" rx="2" fill="rgba(30,30,30,0.85)" display="none"/>' +
                        '<text class="ts-qylabel" x="38" y="0" font-size="8" fill="white" text-anchor="end" alignment-baseline="middle" display="none"/>' +
                        '<rect class="ts-qxlabel-bg" x="0" y="' + (QH - 16) + '" width="48" height="14" rx="2" fill="rgba(30,30,30,0.85)" display="none"/>' +
                        '<text class="ts-qxlabel" x="0" y="' + (QH - 9) + '" font-size="8" fill="white" text-anchor="middle" alignment-baseline="middle" display="none"/>' +
                        '<rect class="ts-qover" x="' + ML + '" y="' + QT + '" width="' + pw + '" height="' + qph + '" fill="transparent"/>' +
                        '</svg>';
                }

                // Percentile dropdown
                const pctOpts = Object.entries(PCT_LABELS).map(([v, l]) =>
                    '<option value="' + v + '"' + (parseFloat(v) === state.pct ? ' selected' : '') + '>' + l + '</option>'
                ).join('');

                // Window size dropdown (show actual size when in auto mode)
                const autoWs = state.wsMode === 'auto' ? state.ws : chooseWinSize(range);
                const autoLabel = autoWs >= 1 ? autoWs + 's' : (autoWs * 1000) + 'ms';
                const wsOpts = [
                    {v: 'auto', l: 'Auto (' + autoLabel + ')'},
                    {v: '0.1', l: '100ms'},
                    {v: '0.25', l: '250ms'},
                    {v: '0.5', l: '500ms'},
                    {v: '1', l: '1s'},
                    {v: '2', l: '2s'},
                    {v: '5', l: '5s'},
                    {v: '10', l: '10s'},
                    {v: '30', l: '30s'},
                    {v: '60', l: '60s'},
                ].map(o => '<option value="' + o.v + '"' + (o.v === state.wsMode ? ' selected' : '') + '>' + o.l + '</option>').join('');

                // Stage legend
                const allStages = ['embed_sparse', 'embed_dense', 'store', 'chunk', 'embed'];
                const sLeg = allStages.filter(s => data.some(d => d.stage === s)).map(sid => {
                    const vis = state.vis.has(sid), p = aggP50(sid);
                    return '<span class="ts-sleg" data-s="' + sid + '" style="cursor:pointer;display:inline-flex;align-items:center;gap:3px;' + (vis ? '' : 'opacity:0.35;text-decoration:line-through;') + '">' +
                        '<span style="width:8px;height:8px;border-radius:50%;background:' + COLORS[sid] + ';display:inline-block"></span>' +
                        LABELS[sid] + (p ? ' ' + p : '') + '</span>';
                }).join('');

                // Queue legend (grouped: waiting → processing → done)
                const qGroups = [
                    {title: 'Queued', items: [
                        {key: 'file_queue', label: 'chunk', dataKey: 'file_queue', dashed: false},
                        {key: 'embed_queue', label: 'embed', dataKey: 'embed_queue', dashed: false},
                        {key: 'upsert_queue', label: 'store', dataKey: 'upsert_queue', dashed: false},
                    ]},
                    {title: 'Processing', items: [
                        {key: 'chunk_inflight', label: 'chunk', dataKey: 'chunk_in_flight', dashed: true},
                        {key: 'embed_inflight', label: 'embed', dataKey: 'embed_in_flight', dashed: true},
                        {key: 'store_inflight', label: 'store', dataKey: 'store_in_flight', dashed: true},
                    ]},
                    {title: 'Completed', items: [
                        {key: 'chunk_done', label: 'chunk', dataKey: 'files_chunk_done', dashed: false},
                        {key: 'embed_done', label: 'embed', dataKey: 'files_embed_done', dashed: false},
                        {key: 'store_done', label: 'store', dataKey: 'files_store_done', dashed: false},
                    ]},
                ];
                const qLeg = qGroups.map(g => {
                    const items = g.items.map(({key, label, dataKey, dashed}) => {
                        const vis = state.qvis.has(key);
                        let peak = 0;
                        if (dataKey === null && key === 'cumulative') {
                            const sd = data.find(d => d.stage === 'store');
                            peak = sd ? sd.completions.length : 0;
                        } else if (dataKey) {
                            peak = Math.max(0, ...queueData.map(s => s[dataKey]));
                        }
                        const dotStyle = dashed
                            ? 'width:8px;height:8px;border-radius:50%;border:2px solid ' + QCOLORS[key] + ';background:transparent;display:inline-block'
                            : 'width:8px;height:8px;border-radius:50%;background:' + QCOLORS[key] + ';display:inline-block';
                        const unit = key === 'cumulative' ? ' files' : ' files pk';
                        return '<span class="ts-qleg" data-q="' + key + '" style="cursor:pointer;display:inline-flex;align-items:center;gap:3px;' + (vis ? '' : 'opacity:0.35;text-decoration:line-through;') + '">' +
                            '<span style="' + dotStyle + '"></span>' +
                            label + ' (' + peak.toLocaleString() + unit + ')</span>';
                    }).join('');
                    const groupKeys = g.items.map(i => i.key).join(',');
                    return (g.title ? '<span class="ts-qgrp" data-keys="' + groupKeys + '" style="cursor:pointer;color:#aaa;font-size:10px;font-weight:600;user-select:none" title="Click to toggle group">' + g.title + ':</span> ' : '') + items;
                }).join(' &nbsp; ');

                const zoomed = state.start > 0.1 || state.end < totalElapsed - 0.1;
                const zoomInfo = zoomed ? '<div style="font-size:10px;color:#888;margin-top:2px">' +
                    'Showing ' + fmtT(state.start) + ' \\u2013 ' + fmtT(state.end) + ' of ' + fmtT(totalElapsed) +
                    ' <span class="ts-reset" style="cursor:pointer;color:#1976d2;text-decoration:underline">Reset</span></div>' : '';

                const scaleLabel = state.logScale ? 'log' : 'linear';
                const heightOpts = Object.keys(HEIGHT_PRESETS).map(k =>
                    '<option value="' + k + '"' + (k === state.heightKey ? ' selected' : '') + '>' + k + '</option>'
                ).join('');

                el.innerHTML =
                    '<div style="font-size:11px;color:#888;display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;padding-left:7.3%">' +
                    '<span>Processing Time (' + scaleLabel + ')</span>' +
                    '<div style="display:flex;gap:6px;align-items:center">' +
                    '<label style="font-size:10px;color:#666;cursor:pointer;display:flex;align-items:center;gap:2px"><input type="checkbox" class="ts-log"' + (state.logScale ? ' checked' : '') + ' style="margin:0"> Log</label>' +
                    '<select class="ts-pct" style="font-size:10px;border:1px solid #ddd;border-radius:3px;padding:1px 4px;color:#666">' + pctOpts + '</select>' +
                    '<select class="ts-ws" style="font-size:10px;border:1px solid #ddd;border-radius:3px;padding:1px 4px;color:#666">' + wsOpts + '</select>' +
                    '<select class="ts-height" style="font-size:10px;border:1px solid #ddd;border-radius:3px;padding:1px 4px;color:#666" title="Chart height">' + heightOpts + '</select>' +
                    '</div></div>' +
                    '<svg viewBox="0 0 ' + W + ' ' + PH + '" preserveAspectRatio="xMidYMid meet" style="width:100%;cursor:crosshair" class="ts-pchart">' +
                    '<rect x="' + ML + '" y="' + PT + '" width="' + pw + '" height="' + ph + '" fill="#fafafa" stroke="' + THEME.colors.border + '" stroke-width="1"/>' +
                    yAxis.svg +
                    '<line x1="' + ML + '" y1="' + (PT + ph) + '" x2="' + (ML + pw) + '" y2="' + (PT + ph) + '" stroke="' + THEME.colors.axis + '" stroke-width="1"/>' +
                    '<line x1="' + ML + '" y1="' + PT + '" x2="' + ML + '" y2="' + (PT + ph) + '" stroke="' + THEME.colors.axis + '" stroke-width="1"/>' +
                    xAxis.svg + paths +
                    '<line class="ts-pcross" x1="0" y1="' + PT + '" x2="0" y2="' + (PT + ph) + '" stroke="#999" stroke-width="0.5" stroke-dasharray="2" display="none"/>' +
                    '<line class="ts-phcross" x1="' + ML + '" y1="0" x2="' + (ML + pw) + '" y2="0" stroke="#999" stroke-width="0.5" stroke-dasharray="2" display="none"/>' +
                    '<rect class="ts-pylabel-bg" x="2" y="0" width="38" height="14" rx="2" fill="rgba(30,30,30,0.85)" display="none"/>' +
                    '<text class="ts-pylabel" x="38" y="0" font-size="8" fill="white" text-anchor="end" alignment-baseline="middle" display="none"/>' +
                    '<rect class="ts-pxlabel-bg" x="0" y="' + (PH - 16) + '" width="48" height="14" rx="2" fill="rgba(30,30,30,0.85)" display="none"/>' +
                    '<text class="ts-pxlabel" x="0" y="' + (PH - 9) + '" font-size="8" fill="white" text-anchor="middle" alignment-baseline="middle" display="none"/>' +
                    '<circle class="ts-phighlight" cx="0" cy="0" r="' + THEME.chart.dotHoverR + '" fill="transparent" stroke="white" stroke-width="1.5" display="none"/>' +
                    '<rect class="ts-sel" x="0" y="' + PT + '" width="0" height="' + ph + '" fill="rgba(25,118,210,0.15)" stroke="#1976d2" stroke-width="0.5" display="none"/>' +
                    '<rect class="ts-pover" x="' + ML + '" y="' + PT + '" width="' + pw + '" height="' + ph + '" fill="transparent"/>' +
                    '</svg>' +
                    '<div style="display:flex;gap:12px;font-size:11px;color:#555;margin:4px 0;flex-wrap:wrap;padding-left:7.3%">' + sLeg +
                    '<span style="color:#bbb;font-size:9px;margin-left:auto">shift-click to solo</span></div>' +
                    zoomInfo +
                    '<div style="font-size:11px;color:#888;margin-top:24px;margin-bottom:8px;padding-left:7.3%">Queue Depth (1Hz)</div>' +
                    qSvg +
                    '<div style="display:flex;gap:12px;font-size:11px;color:#555;margin-top:4px;flex-wrap:wrap;padding-left:7.3%">' + qLeg + '</div>';

                bindEvents();
            }

            function bindEvents() {
                // Percentile
                const pctSel = el.querySelector('.ts-pct');
                if (pctSel) pctSel.onchange = (e) => { state.pct = parseFloat(e.target.value); render(); };

                // Window size
                const wsSel = el.querySelector('.ts-ws');
                if (wsSel) wsSel.onchange = (e) => { state.wsMode = e.target.value; render(); };

                // Log/linear toggle
                const logCb = el.querySelector('.ts-log');
                if (logCb) logCb.onchange = (e) => { state.logScale = e.target.checked; render(); };

                // Height preset
                const htSel = el.querySelector('.ts-height');
                if (htSel) htSel.onchange = (e) => { state.heightKey = e.target.value; render(); };

                // Stage toggle (shift-click to solo)
                el.querySelectorAll('.ts-sleg').forEach(x => {
                    x.onclick = (e) => {
                        const s = x.dataset.s;
                        if (e.shiftKey) {
                            if (state.vis.size === 1 && state.vis.has(s)) {
                                for (const sd of data) state.vis.add(sd.stage);
                            } else {
                                state.vis.clear(); state.vis.add(s);
                            }
                        } else {
                            state.vis.has(s) ? state.vis.delete(s) : state.vis.add(s);
                        }
                        render();
                    };
                });

                // Queue toggle (shift-click to solo)
                el.querySelectorAll('.ts-qleg').forEach(x => {
                    x.onclick = (e) => {
                        const q = x.dataset.q;
                        if (e.shiftKey) {
                            const allQ = ['file_queue', 'embed_queue', 'upsert_queue', 'chunk_inflight', 'embed_inflight', 'store_inflight', 'cumulative'];
                            if (state.qvis.size === 1 && state.qvis.has(q)) {
                                for (const k of allQ) state.qvis.add(k);
                            } else {
                                state.qvis.clear(); state.qvis.add(q);
                            }
                        } else {
                            state.qvis.has(q) ? state.qvis.delete(q) : state.qvis.add(q);
                        }
                        render();
                    };
                });

                // Group toggle (click group title to toggle all items in group)
                el.querySelectorAll('.ts-qgrp').forEach(x => {
                    x.onclick = () => {
                        const keys = x.dataset.keys.split(',');
                        const allOn = keys.every(k => state.qvis.has(k));
                        for (const k of keys) { allOn ? state.qvis.delete(k) : state.qvis.add(k); }
                        render();
                    };
                });

                // Zoom reset
                const reset = el.querySelector('.ts-reset');
                if (reset) reset.onclick = () => { state.start = 0; state.end = totalElapsed; render(); };

                // Chart interactions
                const pOver = el.querySelector('.ts-pover');
                const qOver = el.querySelector('.ts-qover');
                const pCross = el.querySelector('.ts-pcross');
                const qCross = el.querySelector('.ts-qcross');
                const sel = el.querySelector('.ts-sel');

                function svgTime(e, svg) {
                    const r = svg.getBoundingClientRect();
                    const sx = W / r.width;
                    return x2t((e.clientX - r.left) * sx);
                }

                function showCross(t, yPixel, onPChart) {
                    const xp = t2x(t);
                    if (pCross) { pCross.setAttribute('x1', xp); pCross.setAttribute('x2', xp); pCross.setAttribute('display', ''); }
                    if (qCross) { qCross.setAttribute('x1', xp); qCross.setAttribute('x2', xp); qCross.setAttribute('display', ''); }

                    const hp = HEIGHT_PRESETS[state.heightKey];
                    const PH = hp.ph, QH = hp.qh;
                    const ph = PH - PT - PB, qph = QH - QT - QB;

                    const phcross = el.querySelector('.ts-phcross');
                    const pylabelBg = el.querySelector('.ts-pylabel-bg');
                    const pylabel = el.querySelector('.ts-pylabel');
                    const pxlabelBg = el.querySelector('.ts-pxlabel-bg');
                    const pxlabel = el.querySelector('.ts-pxlabel');
                    const qhcross = el.querySelector('.ts-qhcross');
                    const qylabelBg = el.querySelector('.ts-qylabel-bg');
                    const qylabel = el.querySelector('.ts-qylabel');
                    const qxlabelBg = el.querySelector('.ts-qxlabel-bg');
                    const qxlabel = el.querySelector('.ts-qxlabel');

                    // X-axis time label (only on hovered chart)
                    if (onPChart && pxlabel && pxlabelBg) {
                        pxlabelBg.setAttribute('x', xp - 24); pxlabelBg.setAttribute('display', '');
                        pxlabel.setAttribute('x', xp); pxlabel.textContent = fmtT(t); pxlabel.setAttribute('display', '');
                    }
                    if (!onPChart && qxlabel && qxlabelBg) {
                        qxlabelBg.setAttribute('x', xp - 24); qxlabelBg.setAttribute('display', '');
                        qxlabel.setAttribute('x', xp); qxlabel.textContent = fmtT(t); qxlabel.setAttribute('display', '');
                    }

                    if (onPChart && phcross && pylabel && state.y2v) {
                        const yVal = state.y2v(yPixel);
                        phcross.setAttribute('y1', yPixel); phcross.setAttribute('y2', yPixel); phcross.setAttribute('display', '');
                        pylabelBg.setAttribute('y', yPixel - 7); pylabelBg.setAttribute('display', '');
                        pylabel.setAttribute('y', yPixel); pylabel.textContent = fmtMs(yVal); pylabel.setAttribute('display', '');
                    }

                    if (!onPChart && qhcross && qylabel && yPixel >= QT && yPixel <= QT + qph) {
                        const yVal = ((QT + qph - yPixel) / qph) * state.qMaxY;
                        qhcross.setAttribute('y1', yPixel); qhcross.setAttribute('y2', yPixel); qhcross.setAttribute('display', '');
                        qylabelBg.setAttribute('y', yPixel - 7); qylabelBg.setAttribute('display', '');
                        qylabel.setAttribute('y', yPixel); qylabel.textContent = Math.round(yVal).toString(); qylabel.setAttribute('display', '');
                    }
                }

                function hideCross() {
                    const elems = [pCross, qCross, el.querySelector('.ts-phcross'), el.querySelector('.ts-qhcross'),
                                   el.querySelector('.ts-pylabel'), el.querySelector('.ts-pylabel-bg'),
                                   el.querySelector('.ts-pxlabel'), el.querySelector('.ts-pxlabel-bg'),
                                   el.querySelector('.ts-qylabel'), el.querySelector('.ts-qylabel-bg'),
                                   el.querySelector('.ts-qxlabel'), el.querySelector('.ts-qxlabel-bg'),
                                   el.querySelector('.ts-phighlight')];
                    for (const elem of elems) { if (elem) elem.setAttribute('display', 'none'); }
                    tooltip.style.display = 'none';
                }

                function showTip(e, t, onPChart) {
                    const winIdx = Math.floor((t - state.start) / state.ws);
                    const winStart = state.start + winIdx * state.ws;
                    const winEnd = Math.min(state.end, winStart + state.ws);
                    let html = '<div style="color:#aaa">' + fmtT(t) + (onPChart ? ' | window: ' + fmtT(winStart) + '\\u2013' + fmtT(winEnd) : '') + '</div>';

                    if (onPChart) {
                        for (const [sid, wins] of Object.entries(state.wins)) {
                            const idx = Math.floor((t - state.start) / state.ws);
                            const w = wins[idx];
                            if (w && w.v !== null) {
                                html += '<div><span style="color:' + COLORS[sid] + '">\\u25cf</span> ' +
                                    (LABELS[sid] || sid).padEnd(8) + ' ' + fmtMs(w.v).padStart(8) +
                                    ' <span style="color:#888">(' + w.n + ')</span></div>';
                            }
                        }
                    }

                    const qd = queueAt(t);
                    if (qd) {
                        const qparts = [], iparts = [];
                        if (state.qvis.has('file_queue')) qparts.push('chunk ' + qd.file_queue);
                        if (state.qvis.has('embed_queue')) qparts.push('embed ' + qd.embed_queue);
                        if (state.qvis.has('upsert_queue')) qparts.push('store ' + qd.upsert_queue);
                        if (state.qvis.has('chunk_inflight')) iparts.push('chunk ' + qd.chunk_in_flight);
                        if (state.qvis.has('embed_inflight')) iparts.push('embed ' + qd.embed_in_flight);
                        if (state.qvis.has('store_inflight')) iparts.push('store ' + qd.store_in_flight);
                        if (qparts.length) html += '<div style="color:#888;margin-top:2px">Queued: ' + qparts.join('  ') + ' files</div>';
                        if (iparts.length) html += '<div style="color:#888">Processing: ' + iparts.join('  ') + ' files</div>';
                        const cparts = [];
                        if (state.qvis.has('chunk_done') && qd.files_chunk_done !== undefined) cparts.push('chunk ' + qd.files_chunk_done);
                        if (state.qvis.has('embed_done') && qd.files_embed_done !== undefined) cparts.push('embed ' + qd.files_embed_done);
                        if (state.qvis.has('store_done') && qd.files_store_done !== undefined) cparts.push('store ' + qd.files_store_done);
                        if (cparts.length) html += '<div style="color:#888">Completed: ' + cparts.join('  ') + ' files</div>';
                    }
                    tooltip.innerHTML = html;
                    tooltip.style.display = 'block';
                    let tx = e.clientX + 12, ty = e.clientY - 10;
                    tooltip.style.left = tx + 'px';
                    tooltip.style.top = ty + 'px';
                    const tr = tooltip.getBoundingClientRect();
                    if (tr.right > window.innerWidth) tooltip.style.left = (e.clientX - tr.width - 12) + 'px';
                    if (tr.bottom > window.innerHeight) tooltip.style.top = (e.clientY - tr.height - 10) + 'px';
                }

                function onMove(e, svg) {
                    const t = svgTime(e, svg);
                    if (t < state.start || t > state.end) return;

                    const onPChart = svg.classList.contains('ts-pchart');
                    const hp = HEIGHT_PRESETS[state.heightKey];
                    const svgH = onPChart ? hp.ph : hp.qh;
                    const r = svg.getBoundingClientRect();
                    const sx = W / r.width, sy = svgH / r.height;
                    const yPix = (e.clientY - r.top) * sy;

                    showCross(t, yPix, onPChart);

                    if (dragStart !== null && sel) {
                        const x1 = t2x(Math.min(dragStart, t)), x2 = t2x(Math.max(dragStart, t));
                        sel.setAttribute('x', x1); sel.setAttribute('width', x2 - x1); sel.setAttribute('display', '');
                        tooltip.style.display = 'none';
                    } else {
                        const highlight = el.querySelector('.ts-phighlight');
                        if (onPChart && highlight && state.px && state.py) {
                            const winIdx = Math.floor((t - state.start) / state.ws);
                            let nearestDot = null, nearestDist = Infinity;
                            for (const [sid, wins] of Object.entries(state.wins)) {
                                const w = wins[winIdx];
                                if (w && w.v !== null) {
                                    const cx = state.px((w.s + w.e) / 2), cy = state.py(w.v);
                                    const dist = Math.abs(cy - yPix);
                                    if (dist < nearestDist) { nearestDist = dist; nearestDot = {cx, cy, color: COLORS[sid]}; }
                                }
                            }
                            if (nearestDot) {
                                highlight.setAttribute('cx', nearestDot.cx); highlight.setAttribute('cy', nearestDot.cy);
                                highlight.setAttribute('fill', nearestDot.color); highlight.setAttribute('display', '');
                            } else {
                                highlight.setAttribute('display', 'none');
                            }
                        }
                        showTip(e, t, onPChart);
                    }
                }

                function onDown(e, svg) { dragStart = svgTime(e, svg); }

                function onUp(e, svg) {
                    if (dragStart !== null) {
                        const dragEnd = svgTime(e, svg);
                        const zs = Math.min(dragStart, dragEnd), ze = Math.max(dragStart, dragEnd);
                        if (ze - zs > 0.1) {
                            state.start = Math.max(0, zs); state.end = Math.min(totalElapsed, ze); render();
                        }
                    }
                    dragStart = null;
                    if (sel) sel.setAttribute('display', 'none');
                }

                function onLeave() { hideCross(); dragStart = null; if (sel) sel.setAttribute('display', 'none'); }

                function onWheel(e, svg) {
                    e.preventDefault();
                    const t = svgTime(e, svg), range = state.end - state.start;
                    const factor = e.deltaY > 0 ? 1.08 : 0.93;
                    const nr = Math.max(0.5, Math.min(totalElapsed, range * factor));
                    const ratio = (t - state.start) / range;
                    state.start = Math.max(0, t - ratio * nr);
                    state.end = Math.min(totalElapsed, state.start + nr);
                    scheduleRender();
                }

                function onDblClick() { state.start = 0; state.end = totalElapsed; render(); }

                function bind(overlay) {
                    if (!overlay) return;
                    const svg = overlay.closest('svg');
                    overlay.addEventListener('mousemove', (e) => onMove(e, svg));
                    overlay.addEventListener('mousedown', (e) => onDown(e, svg));
                    overlay.addEventListener('mouseup', (e) => onUp(e, svg));
                    overlay.addEventListener('mouseleave', onLeave);
                    overlay.addEventListener('wheel', (e) => onWheel(e, svg), {passive: false});
                    overlay.addEventListener('dblclick', onDblClick);
                }
                bind(pOver);
                bind(qOver);
            }

            render();
        }

        function initPendingCharts() {
            const consumed = new Set();
            document.querySelectorAll('.ts-chart-pending').forEach(el => {
                const d = pendingCharts[el.id];
                if (d) {
                    el.classList.remove('ts-chart-pending');
                    createTimeSeriesChart(el.id, d.series, d.qSeries, d.elapsed, d.stages);
                    consumed.add(el.id);
                }
            });
            for (const id of consumed) delete pendingCharts[id];
        }

        function formatTimingReport(timing, opId) {
            if (!timing || !timing.stages || timing.stages.length === 0) return '';
            const SUB_STAGES = new Set(['embed_dense', 'embed_sparse']);
            const fmtMs = (v) => v < 1 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : Math.round(v).toLocaleString();
            const fmtPct = (stats) => stats ? `${fmtMs(stats.p50_ms)} / ${fmtMs(stats.p95_ms)} / ${fmtMs(stats.p99_ms)} / ${fmtMs(stats.max_ms)}` : '—';

            const threads = timing.sparse_threads;
            let rows = timing.stages.map(s => {
                const isSub = SUB_STAGES.has(s.stage);
                const label = s.stage.replace('embed_', '↳ ');
                // Efficiency: mean cpu / mean wall / threads (only for embed_sparse)
                let effHtml = '—';
                if (s.cpu && s.wall && threads && s.wall.p50_ms > 0) {
                    const parallel = s.cpu.p50_ms / s.wall.p50_ms;
                    const eff = parallel / threads;
                    effHtml = `${parallel.toFixed(1)}x / ${threads}t = ${(eff * 100).toFixed(0)}%`;
                }
                return `<tr class="${isSub ? 'sub-stage' : ''}">
                    <td>${label}</td>
                    <td>${fmtPct(s.processing)}</td>
                    <td>${s.queue_wait ? fmtPct(s.queue_wait) : '—'}</td>
                    <td>${s.batch_wait ? fmtPct(s.batch_wait) : '—'}</td>
                    <td>${s.cpu ? fmtPct(s.cpu) : '—'}</td>
                    <td>${s.wall ? fmtPct(s.wall) : '—'}</td>
                    <td>${effHtml}</td>
                    <td>${s.throughput_per_sec.toFixed(1)}/s</td>
                    <td>${s.processing.count.toLocaleString()}</td>
                </tr>`;
            }).join('');

            let chartHtml;
            if (timing.completion_series && timing.completion_series.length > 0 && opId) {
                const chartId = 'tschart-' + opId;
                pendingCharts[chartId] = {
                    series: timing.completion_series,
                    qSeries: timing.queue_depth_series,
                    elapsed: timing.total_elapsed_seconds,
                    stages: timing.stages,
                };
                chartHtml = '<div id="' + chartId + '" class="ts-chart-pending" style="margin-top:12px"></div>';
            } else {
                chartHtml = formatQueueDepthChart(timing.queue_depth_series);
            }

            return `
                <div style="margin-top: 12px;">
                    <table class="timing-table">
                        <tr>
                            <th>Stage</th>
                            <th>Processing p50/p95/p99/max (ms)</th>
                            <th>Queue Wait</th>
                            <th>Batch Wait</th>
                            <th>CPU p50/p95/p99/max (ms)</th>
                            <th>Wall p50/p95/p99/max (ms)</th>
                            <th>Efficiency</th>
                            <th>Throughput</th>
                            <th>Items</th>
                        </tr>
                        ${rows}
                    </table>
                    <div style="font-size:11px;color:#888;margin-top:4px">
                        Scan: ${timing.scan_seconds.toFixed(1)}s | ${timing.total_items.toLocaleString()} items | ${timing.total_elapsed_seconds.toFixed(1)}s total${timing.sparse_threads ? ` | ${timing.sparse_threads} rayon threads` : ''}
                    </div>
                    ${chartHtml}
                </div>`;
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
        const pendingCharts = {};

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
                const resp = await fetch(`/api/operations/${opId}/logs?tail=1000`);
                const data = await resp.json();
                const el = document.getElementById('logs-' + opId);
                const countEl = document.getElementById('log-count-' + opId);
                if (!el) return;

                // Skip re-render if log file hasn't changed
                if (lastLogLines[opId] === data.total_lines) return;
                lastLogLines[opId] = data.total_lines;

                countEl.textContent = data.lines.length < data.total_lines
                    ? `showing ${data.lines.length} of ${data.total_lines} lines`
                    : `${data.total_lines} lines`;
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
                }).join('\\n');
                el.innerHTML = `<div class="log-viewer">${html}</div>`;
                // Auto-scroll to bottom
                el.querySelector('.log-viewer').scrollTop = el.querySelector('.log-viewer').scrollHeight;
            } catch (e) {
                console.error('Log fetch failed:', e);
            }
        }

        // Track state to avoid unnecessary DOM replacement
        let lastActiveIds = '';
        let lastRecentIds = '';
        let lastServerCount = -1;
        const lastLogLines = {};  // opId -> total_lines (skip re-render when unchanged)

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

                // Active operations — targeted update to preserve open logs
                const activeIds = activeOps.map(o => o.operation_id).join(',');
                if (activeIds !== lastActiveIds) {
                    lastActiveIds = activeIds;
                    document.getElementById('active-ops').innerHTML =
                        activeOps.length === 0
                            ? '<div class="empty">No active operations</div>'
                            : activeOps.map(formatOperation).join('');
                    // Re-expand logs after full re-render
                    for (const opId of openLogs) {
                        const el = document.getElementById('logs-' + opId);
                        if (el && activeOps.some(o => o.operation_id === opId)) {
                            el.style.display = 'block';
                            const toggle = document.getElementById('log-toggle-' + opId);
                            if (toggle) toggle.textContent = '▼ Logs';
                            fetchLogs(opId);
                        }
                    }
                } else if (activeOps.length > 0) {
                    // Same ops — update only progress and meta, leave logs untouched
                    for (const op of activeOps) {
                        const metaEl = document.getElementById('meta-' + op.operation_id);
                        const progressEl = document.getElementById('progress-' + op.operation_id);
                        if (metaEl) metaEl.innerHTML = formatOperationMeta(op);
                        if (progressEl) progressEl.innerHTML = formatOperationProgressHtml(op);
                    }
                    // Refresh open logs (fetchLogs skips if unchanged)
                    for (const opId of openLogs) {
                        if (activeOps.some(o => o.operation_id === opId)) fetchLogs(opId);
                    }
                }

                // Servers — only re-render when count changes
                if (servers.length !== lastServerCount) {
                    lastServerCount = servers.length;
                    document.getElementById('servers').innerHTML =
                        servers.length === 0
                            ? '<div class="empty">No MCP servers connected</div>'
                            : servers.map(s =>
                                `<div class="server">PID: <strong>${s.pid}</strong> | Started: ${new Date(s.started_at).toLocaleString()}</div>`
                            ).join('');
                }

                // Recent completed — only re-render when operations are added/removed
                const completedOps = recentOps.filter(o => o.ended_at);
                const recentIds = completedOps.map(o => o.operation_id).join(',');
                document.getElementById('btn-clear-all').disabled = completedOps.length === 0;

                if (recentIds !== lastRecentIds) {
                    lastRecentIds = recentIds;
                    document.getElementById('recent-ops').innerHTML =
                        completedOps.length === 0
                            ? '<div class="empty">No recent operations</div>'
                            : completedOps.slice(0, 5).map(formatOperation).join('');
                    // Re-expand logs after DOM replacement
                    for (const opId of openLogs) {
                        const el = document.getElementById('logs-' + opId);
                        const toggle = document.getElementById('log-toggle-' + opId);
                        if (el) {
                            el.style.display = 'block';
                            if (toggle) toggle.textContent = '▼ Logs';
                            fetchLogs(opId);
                        } else {
                            openLogs.delete(opId);
                        }
                    }
                }

                initPendingCharts();
            } catch (e) {
                console.error('Update failed:', e);
            }
        }

        async function restartDashboard() {
            try {
                await fetch('/api/restart', { method: 'POST' });
                document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;color:#666;"><div style="text-align:center;"><h2>Restarting dashboard...</h2><p>Reconnecting automatically</p></div></div>';
                // Poll until new dashboard is up
                const tryReconnect = async () => {
                    for (let i = 0; i < 30; i++) {
                        await new Promise(r => setTimeout(r, 500));
                        try {
                            const resp = await fetch('/api/state');
                            if (resp.ok) { location.href = '/?v=' + Date.now(); return; }
                        } catch (e) { /* still restarting */ }
                    }
                    document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;font-family:sans-serif;color:#c62828;"><h2>Dashboard failed to restart. Check MCP server.</h2></div>';
                };
                tryReconnect();
            } catch (e) {
                console.error('Restart failed:', e);
            }
        }

        async function deleteOp(opId) {
            await fetch(`/api/operations/${opId}`, { method: 'DELETE' });
            openLogs.delete(opId);
            update();
        }

        async function clearAllOps() {
            await fetch('/api/operations', { method: 'DELETE' });
            openLogs.clear();
            update();
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
            file_path = _operation_path(operation_id, '.json')
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
            log_path = _operation_path(operation_id, '.log')
            if not log_path.exists():
                return {'lines': [], 'total_lines': 0}
            text = log_path.read_text()
            all_lines = text.splitlines()
            return {
                'lines': all_lines[-tail:],
                'total_lines': len(all_lines),
            }

        @app.delete('/api/operations/{operation_id}')
        def delete_operation(operation_id: str) -> Mapping[str, bool]:
            """Delete a single operation and its log."""
            json_path = _operation_path(operation_id, '.json')
            log_path = _operation_path(operation_id, '.log')
            if not json_path.exists():
                raise HTTPException(status_code=404, detail='Operation not found')
            json_path.unlink()
            if log_path.exists():
                log_path.unlink()
            return {'deleted': True}

        @app.delete('/api/operations')
        def clear_all_operations() -> Mapping[str, int]:
            """Clear all completed operations."""
            count = 0
            if OPERATIONS_DIR.exists():
                for f in OPERATIONS_DIR.glob('*.json'):
                    # Skip active operations
                    data = json.loads(f.read_text())
                    if data.get('ended_at') is None:
                        continue
                    op_id = f.stem
                    f.unlink()
                    log_path = OPERATIONS_DIR / f'{op_id}.log'
                    if log_path.exists():
                        log_path.unlink()
                    count += 1
            return {'cleared': count}

        @app.post('/api/restart')
        def restart_dashboard(background_tasks: BackgroundTasks) -> Mapping[str, str]:
            """Restart dashboard to pick up code changes.

            Spawns a new dashboard process then exits this one.
            The new process retries the same port (with delay for OS release).
            """
            subprocess.Popen(
                [sys.executable, '-m', 'document_search.dashboard'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            background_tasks.add_task(_exit_for_restart)
            return {'status': 'restarting'}

        @app.get('/', response_class=HTMLResponse)
        def index() -> str:
            return INDEX_HTML

        return app


def _find_port(preferred: int | None, *, retries: int = 10, retry_delay: float = 0.2) -> int:
    """Find available port. Retries preferred port before falling back.

    When restarting, the OS may not have released the port yet.
    Retries with short delays give the kernel time to reclaim it.
    """
    if preferred and _port_available(preferred):
        return preferred

    # Retry preferred port (handles restart race)
    if preferred:
        for _ in range(retries):
            time.sleep(retry_delay)
            if _port_available(preferred):
                return preferred

    # Fall back to default, then OS-assigned
    if preferred != DEFAULT_PORT and _port_available(DEFAULT_PORT):
        return DEFAULT_PORT

    return _get_free_port()


def _port_available(port: int) -> bool:
    """Check if port is available (handles TIME_WAIT from recent restarts)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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


def _exit_for_restart() -> None:
    """Hard-exit to release the port for the new dashboard process.

    Bypasses finally blocks intentionally — immediate port release is required.
    New process reads existing MCP registrations from state file on disk.
    """
    os._exit(0)


_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')


def _operation_path(operation_id: str, suffix: str) -> Path:
    """Resolve operation file path with traversal protection."""
    if not _UUID_RE.match(operation_id):
        raise HTTPException(status_code=400, detail='Invalid operation ID')
    return OPERATIONS_DIR / f'{operation_id}{suffix}'


def _read_operations() -> Sequence[OperationState]:
    """Read all operation files from disk, sorted by created_at descending."""
    if not OPERATIONS_DIR.exists():
        return []

    ops: list[OperationState] = []
    for file_path in OPERATIONS_DIR.glob('*.json'):
        data = json.loads(file_path.read_text())
        ops.append(OperationState.model_validate(data))

    return sorted(ops, key=lambda o: o.created_at, reverse=True)
