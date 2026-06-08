# Context

Machine-local context layer. Data is gitignored and never leaves the machine; only the mechanism that maintains it is tracked.

One context per machine, scoped by the account logged into the producing app.

| Directory  | Contents                                                               |
|------------|------------------------------------------------------------------------|
| `granola/` | Granola meeting archive (data-only) — sync via `/sync-granola-context` |
| `notion/`  | Notion workspace export mirror (data-only) — `scripts/process-notion-export.py` |

Each sync indexes the archive for semantic search: `document-search search "<query>" -c document-chunks`.
