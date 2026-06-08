# Notion Export Archive

Machine-local mirror of a Notion workspace for semantic search. Data is gitignored; only this README and the processor mechanism are tracked. Populated by `scripts/process-notion-export.py`.

> This repo is **public** — keep this README workspace-agnostic. Page titles, teamspace names, and the export data itself are machine-local (gitignored); do not name specific workspaces, projects, or internal systems here.

## Vocabulary

Notion nests **workspace → teamspace → page → subpage**. On disk, each teamspace becomes a top-level directory, mirroring the app:

```
context/notion/{workspace-slug}/
  {Teamspace}/{page} {pageid}/{subpage} {pageid}.md      # hierarchy preserved
  index.json                                             # id → title, path, notion.so URL
```

## Two export granularities (the core tradeoff)

| | Workspace-level | Page-level |
|---|---|---|
| Trigger | Settings → "Export all content" (**browser UI only** — no API) | `POST /v1/enqueueTask` (**headless API**, `recursive:true`) |
| Jobs for full backfill | **1 per workspace** | one per top-level page (dozens) |
| Progress telemetry | none (opaque spinner) | `getTasks` poll + live page/file counter |
| Delivery | email only, **no auto-download** | auto-downloads **and** emails |
| Incremental-friendly | no — all-or-nothing | yes — re-export only changed roots |

**Architecture: backfill via workspace export; incremental via page-level API.** A full refresh re-runs the workspace export. To sync only changes, diff `last_edited_time` (official API or the private session API) against the archive, then page-level-export just the changed roots. *(Incremental is not yet built — known mechanism, follow-up.)*

## Acquisition (browser-gated)

The workspace export must be driven through a logged-in browser (selenium-browser, profile-state import):
1. Settings → General → Export → **Markdown & CSV · Everything · Include subpages**.
2. Notion emails a link from **"Notion Team" `<notify@mail.notion.so>`**, subject **"Your workspace export is ready"** (page exports use `export-noreply@` / "Your Notion export is ready" — watch *both*).
3. The link is a **Mailgun click-tracking redirect** (`mg.mail.notion.so/c/…`) — a bare GET / `download-resource` **403s**; only a real browser click resolves it (carries the session through the redirect to the signed `file.notion.com` URL).

## Processing & layout decisions

`process-notion-export.py` unwraps the double-nested zip (outer → `Part-N` inner), strips the single `Export-{id}/` wrapper, and **streams each file to a sanitized path** — preserving Notion's hierarchy (for path-scoped search) while capping path components below the macOS 255-byte filename limit. The `{title} {32-hex-id}` filename suffix is the page id; `index.json` parses it for the `notion.so` backlink.

- **Index text, keep media as assets.** In testing, a multi-GB workspace export was ~99% binary by bytes (video / images / large JSON dumps) and only ~22 MB of markdown across ~14K pages. document-search indexes the text/PDF/CSV; the binaries stay on disk as on-demand assets (hand an image to Claude to read).
- **Scoping comes from the filesystem** (path filtering on the teamspace dirs). If finer/metadata scoping is ever needed, the extension point is frontmatter-metadata indexing in document-search — deferred (YAGNI).

## Indexing

```bash
document-search index context/notion/{slug} -c document-chunks --no-gitignore
document-search search "<query>" -c document-chunks -p context/notion/{slug}/{Teamspace}
```

## Notes

- **mypy exclude:** `context/` is excluded from mypy (in `pyproject.toml`) because Notion exports embed third-party `.py` snippets that would otherwise fail `mypy .`. mypy has no per-directory exclude file (unlike `.gitignore`), so the rationale lives here.
- No partitioning observed even at multi-GB / ~17K files — single `Part-1` (page exports use an `ExportBlock-` filename prefix, workspace exports `Export-`).
- A single recursive root can be enormous — one hub page in testing was ~40% of the whole workspace.
- Workspace export takes tens of minutes to generate and needs **owner/admin** rights (member-level settings expose only "Leave workspace", no Export).
