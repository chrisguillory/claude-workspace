"""Type stubs for pdfplumber.

pdfplumber has inline types but py.typed is missing from wheel distribution.
See: https://github.com/jsvine/pdfplumber/issues/698
"""

from typing import Any

class Page:
    def extract_tables(self) -> list[list[list[str | None]]]: ...

class PDF:
    pages: list[Page]
    def __enter__(self) -> PDF: ...
    def __exit__(self, *args: Any) -> None: ...
    def close(self) -> None: ...

def open(
    path_or_fp: str | Any,
    pages: list[int] | None = None,
    laparams: dict[str, Any] | None = None,
    precision: float = 0.001,
    password: str = '',
) -> PDF: ...
