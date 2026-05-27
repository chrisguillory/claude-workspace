"""Case-folded matching with smart-quote/NBSP fold for user-typed vs. system-stored names."""

from __future__ import annotations

__all__ = [
    'nfkc_casefold',
]

import unicodedata

_QUOTE_FOLD_TABLE = str.maketrans(
    {
        chr(0x2018): "'",  # left single quote
        chr(0x2019): "'",  # right single quote / apostrophe (Core Audio uses this for renamed devices)
        chr(0x201C): '"',  # left double quote
        chr(0x201D): '"',  # right double quote
        chr(0x00A0): ' ',  # NBSP
    }
)


def nfkc_casefold(s: str) -> str:
    """NFKC-normalize, fold smart quotes and NBSP to ASCII, then case-fold.

    The smart-quote fold is the load-bearing piece: Core Audio stores
    user-renamed devices with curly apostrophes (U+2019, e.g.
    ``Chris's AirPods Max``) but keyboards type straight (U+0027). NFKC
    alone does not bridge those — they are different semantic characters,
    not Unicode-equivalence variants.
    """
    return unicodedata.normalize('NFKC', s).translate(_QUOTE_FOLD_TABLE).casefold()
