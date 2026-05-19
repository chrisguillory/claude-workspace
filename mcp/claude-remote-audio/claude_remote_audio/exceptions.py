"""Exception hierarchy for claude-remote-audio.

Structural hierarchy and resolvability are orthogonal axes — the hierarchy
classifies the failure, the ``ResolvableError`` mixin grants structured
recovery context on opt-in subclasses.

AudioError                                       # structural root
└── ApplyError                                   # apply config/constraint violations
    └── ResolvableApplyError(_, ResolvableError) # opt-in: carries code + suggestions

Catch ``AudioError`` for "any failure from this MCP." Catch ``ResolvableError``
(via ``cc_lib.exceptions``) for "any error carrying structured-recovery context,
regardless of source MCP" — the renderer in ``cc_lib.error_boundary`` does
exactly that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cc_lib.exceptions import ResolvableError

__all__ = [
    'ApplyError',
    'AudioError',
    'ResolvableApplyError',
]


class AudioError(Exception):
    """Structural root for claude-remote-audio failures.

    Plain ``Exception`` subclass — no required fields. Use ``except AudioError``
    to catch any failure originating in this MCP. Specific subclasses add
    classification (``ApplyError``) and optional capabilities (``ResolvableError``
    via multiple inheritance, see ``ResolvableApplyError``).
    """


class ApplyError(AudioError):
    """Configuration or constraint violation that prevents ``apply`` from running.

    Default for raise sites that just need a human-readable message — no code,
    no suggestions, no machine-readable structure. Use this for "user passed
    bad input" / "precondition not met" / "discovered topology can't satisfy
    request" kinds of failures.
    """


class ResolvableApplyError(ApplyError, ResolvableError):
    """``ApplyError`` that ALSO carries structured recovery context.

    Use when the failure mode is recognizable (recurring pattern with a stable
    ``code``) AND actionable (inline ``suggestions``, ``docs_url``, or both).
    The CoreAudio HAL wedge is the canonical example — same recognizable
    failure across runs, same recovery procedure (kill coreaudiod, then reboot).

    Caught by ``except ApplyError`` (orchestrator-side) AND
    ``isinstance(exc, ResolvableError)`` (renderer-side) — opts into both
    contracts without ambiguity. Constructor takes positional ``message``
    + keyword-only ``code`` / ``title`` / ``suggestions`` / ``docs_url`` /
    ``context``; see ``ResolvableError`` for field semantics.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        title: str | None = None,
        suggestions: Sequence[str] = (),
        docs_url: str | None = None,
        context: Mapping[str, str] | None = None,
    ) -> None:
        ResolvableError.__init__(
            self,
            message,
            code=code,
            title=title,
            suggestions=suggestions,
            docs_url=docs_url,
            context=context,
        )
