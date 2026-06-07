"""Full-page screenshot via scroll-and-stitch."""

from __future__ import annotations

__all__ = [
    'capture_full_page',
]

import asyncio
import base64
import io
from collections.abc import Awaitable, Callable

from PIL import Image
from selenium import webdriver

# Network-idle waiter supplied by the caller (e.g. BrowserService.wait_for_network_idle) so this
# module needs only the driver, not the service.
type AwaitNetworkIdle = Callable[[int], Awaitable[object]]

# Per-tile dwell before capture: the lazy value lets a re-deferred cross-origin iframe re-render
# while in view; the fast value only lets paint settle.
_TILE_DWELL_LAZY_S = 1.2
_TILE_DWELL_FAST_S = 0.2
_TILE_OVERLAP_PX = 8  # tile overlap so stitched boundaries are seamless

# Flatten sticky/fixed positioning (so it doesn't duplicate across tiles) and hide scrollbars (so they
# don't bake into every tile), stashing each element's original inline position for restoration.
_PREPARE_STITCH_JS = """
document.documentElement.style.scrollBehavior = 'auto';
const style = document.createElement('style');
style.id = '__mcp_stitch_style';
style.textContent = '*{scrollbar-width:none!important}*::-webkit-scrollbar{display:none!important;width:0!important;height:0!important}';
document.head.appendChild(style);
let count = 0;
for (const el of document.querySelectorAll('body *')) {
    const pos = getComputedStyle(el).position;
    if (pos === 'fixed' || pos === 'sticky') {
        el.setAttribute('data-mcp-stitch-pos', el.style.position);
        el.style.setProperty('position', 'static', 'important');
        count++;
    }
}
return count;
"""

# Undo _PREPARE_STITCH_JS: drop the scrollbar-hide style, restore positions, return to the top.
_RESTORE_STITCH_JS = """
const style = document.getElementById('__mcp_stitch_style');
if (style) style.remove();
for (const el of document.querySelectorAll('[data-mcp-stitch-pos]')) {
    const original = el.getAttribute('data-mcp-stitch-pos');
    if (original) { el.style.position = original; } else { el.style.removeProperty('position'); }
    el.removeAttribute('data-mcp-stitch-pos');
}
document.documentElement.style.scrollBehavior = '';
window.scrollTo(0, 0);
"""


async def capture_full_page(
    driver: webdriver.Chrome, *, trigger_lazy: bool, await_network_idle: AwaitNetworkIdle
) -> bytes:
    """Full-page PNG via scroll-and-stitch.

    A single CDP captureBeyondViewport spans the full height but never moves the visual viewport, so
    lazy cross-origin (out-of-process) iframe content — GitHub's Mermaid/math in viewscreen iframes —
    comes back blank, and GitHub re-defers those iframes once scrolled away. So we capture overlapping
    viewport tiles, each while in view, and stitch. Sticky/fixed elements are flattened and scrollbars
    hidden first (so they neither duplicate across tiles nor bake a scrollbar into each), and the page
    is restored afterward. ``trigger_lazy`` dwells the page through first so the deferred iframes render.
    """
    await asyncio.to_thread(driver.execute_script, _PREPARE_STITCH_JS)
    try:
        if trigger_lazy:
            await _warm_up_lazy_content(driver, await_network_idle)
        dwell = _TILE_DWELL_LAZY_S if trigger_lazy else _TILE_DWELL_FAST_S
        return await _stitch_viewport_tiles(driver, dwell)
    finally:
        await asyncio.to_thread(driver.execute_script, _RESTORE_STITCH_JS)


async def _warm_up_lazy_content(driver: webdriver.Chrome, await_network_idle: AwaitNetworkIdle) -> None:
    """Scroll the full page so lazy / deferred content renders before the tile pass.

    Content that renders only when scrolled into view — GitHub's Mermaid/math (deferred cross-origin
    iframe enrichment), lazy-loaded images — stays blank otherwise. Step top→bottom dwelling at each
    stop so IntersectionObservers fire and deferred iframes start loading, then wait for the page to
    settle before returning to the top.
    """
    metrics = await asyncio.to_thread(
        driver.execute_script,
        'return {height: document.body.scrollHeight, vh: window.innerHeight};',
    )
    total_height = int(metrics['height'])
    vh = int(metrics['vh'])
    # Half-viewport stride overlaps stops so every observer fires; the final stop lands on the bottom.
    # Dwell at each stop long enough for the observer callback + GitHub's enrichment to fire (a
    # sub-200ms dwell is too fast — the deferred iframe never starts loading).
    stride = max(vh // 2, 1)
    for y in range(0, total_height + stride, stride):
        await asyncio.to_thread(driver.execute_script, 'window.scrollTo(0, arguments[0]);', min(y, total_height))
        await asyncio.sleep(0.4)
    await _await_page_settled(driver, await_network_idle)
    await asyncio.to_thread(driver.execute_script, 'window.scrollTo(0, 0);')
    await asyncio.sleep(0.3)


async def _await_page_settled(
    driver: webdriver.Chrome, await_network_idle: AwaitNetworkIdle, timeout_s: float = 25.0
) -> None:
    """Block until lazy content finishes loading/rendering, or ``timeout_s`` elapses.

    Deferred content keeps mutating the page after the scroll: cross-origin enrichment iframes
    (Mermaid/geo/3D) load and resize, lazy images decode, KaTeX math typesets. Poll a cheap
    signature — scroll height, loaded-image count, sized-iframe count — and return once it holds
    steady across consecutive checks with the network idle. GitHub keeps its
    ``js-render-needs-enrichment`` class even after rendering, so iframe *height* (folded into the
    signature) is the reliable "done" signal, not that class.
    """
    signature_js = (
        'return document.body.scrollHeight + ":"'
        ' + [...document.querySelectorAll("img")].filter(i => i.complete).length + ":"'
        ' + [...document.querySelectorAll("iframe")].filter(f => f.clientHeight > 0).length;'
    )
    deadline = asyncio.get_running_loop().time() + timeout_s
    previous: str | None = None
    stable = 0
    while asyncio.get_running_loop().time() < deadline:
        await await_network_idle(2000)
        signature = await asyncio.to_thread(driver.execute_script, signature_js)
        if signature == previous:
            stable += 1
            if stable >= 2:
                return
        else:
            stable = 0
            previous = signature
        await asyncio.sleep(0.7)


async def _stitch_viewport_tiles(driver: webdriver.Chrome, tile_dwell_s: float) -> bytes:
    """Scroll the page in overlapping viewport tiles, capture each in view, and stitch into one PNG.

    Each tile dwells ``tile_dwell_s`` before capture so a re-deferred cross-origin iframe re-renders
    while in view; ``captureBeyondViewport=False`` captures only the live viewport — the one path that
    includes out-of-process iframe content. Tiles overlap by ``_TILE_OVERLAP_PX`` so boundaries are
    seamless; the final tile snaps to the bottom.
    """
    metrics = await asyncio.to_thread(
        driver.execute_script,
        'return {w: document.documentElement.clientWidth, h: document.documentElement.scrollHeight,'
        ' vh: window.innerHeight, dpr: window.devicePixelRatio};',
    )
    width = int(metrics['w'])
    total_height = int(metrics['h'])
    viewport_height = int(metrics['vh'])
    dpr = float(metrics['dpr'])
    canvas = Image.new('RGB', (round(width * dpr), round(total_height * dpr)), 'white')
    offset = 0
    while True:
        scroll_y = min(offset, max(0, total_height - viewport_height))
        await asyncio.to_thread(driver.execute_script, 'window.scrollTo(0, arguments[0]);', scroll_y)
        await asyncio.sleep(tile_dwell_s)
        shot = await asyncio.to_thread(
            driver.execute_cdp_cmd,
            'Page.captureScreenshot',
            {'format': 'png', 'fromSurface': True, 'captureBeyondViewport': False},
        )
        tile = Image.open(io.BytesIO(base64.b64decode(shot['data']))).convert('RGB')
        canvas.paste(tile, (0, round(scroll_y * dpr)))
        if scroll_y >= total_height - viewport_height:
            break
        offset += viewport_height - _TILE_OVERLAP_PX
    buffer = io.BytesIO()
    canvas.save(buffer, format='PNG')
    return buffer.getvalue()
