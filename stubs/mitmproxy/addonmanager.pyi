from __future__ import annotations

from typing import Any

class Loader:
    def add_option(self, name: str, typespec: type, default: Any, help: str) -> None: ...
