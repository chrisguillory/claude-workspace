#!/usr/bin/env -S uv run --quiet --script
# /// script
# dependencies = []
# ///
"""Debug hook to print all hook input fields."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Read from stdin
hook_input = json.load(sys.stdin)

# Print each key-value pair to stdout
for key, value in hook_input.items():
    print(f'{key}: {value}')

# Second print to work around output buffering issue
print('')

# Write to debug.txt
try:
    debug_file = Path(hook_input['cwd']) / 'debug.txt'
    with open(debug_file, 'a') as f:
        f.write(f'\n--- {hook_input.get("hook_event_name", "Hook")} ---\n')
        for key, value in hook_input.items():
            f.write(f'{key}: {value}\n')
except Exception as e:
    # Print exception but don't fail the hook
    print(f'Error writing to debug.txt: {e}', file=sys.stderr)
