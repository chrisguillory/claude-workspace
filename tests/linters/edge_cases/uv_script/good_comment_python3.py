#!/usr/bin/env -S uv run --no-project --script
"""Good shebang — python3 mentioned in comment should not trigger."""

# Old shebang was: #!/usr/bin/env python3
from __future__ import annotations

x = 1
