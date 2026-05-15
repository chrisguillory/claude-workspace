#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "mitmproxy>=12",
# ]
# ///

"""Deterministic 429 injection over mitmproxy for binary-patch red/green tests.

Injects ``rate_limit_error`` responses for the first N Anthropic
``/v1/messages`` requests, then passes through. Used to validate the
``force-429-retry-{status,header}`` binary patches without relying on
Anthropic-side server load.

Usage:

    # Terminal 1 — proxy
    scripts/inject-429.py --count 3

    # Terminal 2 — Claude
    export HTTPS_PROXY=http://localhost:8080
    export SSL_CERT_FILE="$HOME/.mitmproxy/mitmproxy-ca-cert.pem"
    claude

After the first run mitmproxy creates a CA at ``~/.mitmproxy/``. Trust it in
the macOS keychain once:

    sudo security add-trusted-cert -d -r trustRoot \\
      -k /Library/Keychains/System.keychain \\
      ~/.mitmproxy/mitmproxy-ca-cert.pem
"""

from __future__ import annotations

import argparse
import sys

from mitmproxy import http
from mitmproxy.tools.main import mitmdump


class Inject429:
    """Return a 429 ``rate_limit_error`` for the first N calls to ``/v1/messages``."""

    def __init__(self, count: int) -> None:
        self._remaining = count

    def request(self, flow: http.HTTPFlow) -> None:
        if 'api.anthropic.com' not in flow.request.host:
            return
        if not flow.request.path.startswith('/v1/messages'):
            return
        if self._remaining <= 0:
            return
        self._remaining -= 1
        flow.response = http.Response.make(
            429,
            (
                b'{"type":"error","error":{"type":"rate_limit_error",'
                b'"message":"Server is temporarily limiting requests '
                b'(not your usage limit) \xc2\xb7 Rate limited"}}'
            ),
            {
                'Content-Type': 'application/json',
                'x-should-retry': 'true',
                'Retry-After': '1',
            },
        )
        print(f'[inject-429] injected 429 ({self._remaining} remaining)', file=sys.stderr)


addons = [Inject429(count=int(__import__('os').environ.get('INJECT_429_COUNT', '3')))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=3, help='Number of 429s to inject before passing through')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    __import__('os').environ['INJECT_429_COUNT'] = str(args.count)
    print(f'[inject-429] injecting first {args.count} /v1/messages requests, listening on :{args.port}', file=sys.stderr)
    mitmdump(['-s', __file__, '-p', str(args.port), '--ssl-insecure'])


if __name__ == '__main__':
    main()
