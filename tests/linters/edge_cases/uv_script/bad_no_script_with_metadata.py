#!/usr/bin/env -S uv run --no-project
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "pydantic",
# ]
# ///
"""Bad shebang — has --no-project with PEP 723 metadata but missing --script."""
