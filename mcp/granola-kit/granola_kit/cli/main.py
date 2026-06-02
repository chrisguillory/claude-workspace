from __future__ import annotations

from cc_lib.cli import add_completion_command, add_help_command, create_app, run_app
from cc_lib.error_boundary import ErrorBoundary

app = create_app(help='granola-kit — Granola.ai meeting notes, transcripts, and chat-based Q&A.')
add_completion_command(app)
add_help_command(app)
error_boundary = ErrorBoundary(exit_code=1)


@error_boundary
def main() -> None:
    """Run the granola-kit CLI."""
    run_app(app)
