# Context

Machine-local context layer. Data is gitignored and never leaves the machine; only the mechanism that maintains it is tracked.

One context per machine, scoped by the account logged into the producing app (e.g., M2 = work Granola account, M5 = personal).

| Directory  | Contents                                                               |
|------------|------------------------------------------------------------------------|
| `granola/` | Granola meeting archive (data-only) — sync via `/sync-granola-context` |

Each sync indexes the archive for semantic search: `document-search search "<query>" -c document-chunks`.
