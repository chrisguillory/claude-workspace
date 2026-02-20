"""Type stubs for bashlex.ast.

bashlex uses a single ``node`` class with different attributes per ``kind``.
All attributes are declared here for permissive access — runtime code uses
``hasattr()``/``getattr()`` to handle per-kind variation. See task #13 for
planned type-safe cleanup.
"""

import builtins

class node:
    kind: str
    pos: tuple[int, int]
    # command, list, pipeline, compound, word, assignment nodes
    parts: builtins.list[node]
    # compound nodes (subshell, brace group)
    list: builtins.list[node]
    redirects: builtins.list[node]
    # word and assignment nodes
    word: str
    # parameter nodes
    value: str
    # redirect nodes
    output: int | node
    type: str
    heredoc: node
    # operator nodes
    op: str
