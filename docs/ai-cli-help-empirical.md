# AI-native CLI help: empirical comparison

## Question

How should `selenium-browser` (and similar Typer CLIs in this workspace) expose
their full surface — every subcommand, every flag, every type, every default,
every example — to AI consumers in **one shot**, so a 1M-context agent can load
the whole API and reason locally without round-tripping through `--help` per
subcommand?

This document is empirical. Every output below was generated against the live
`selenium-browser` CLI and MCP server in this workspace.

---

## The three candidate surfaces, by the numbers

| Surface | How produced | Bytes | Lines | Cost to invoke |
|---|---|---|---|---|
| **Recursive `--help` dump** | Concatenate `selenium-browser <cmd> --help` for all 34 subcommands | 67 KB | 660 | ~5s (one fork+exec per subcommand) |
| **`Click.Context.to_info_dict()` JSON** | One Python call against the live `app` object | 107 KB | 3,940 | ~50 ms |
| **`fastmcp list selenium-browser --json`** | MCP wire call to running server, returns JSON Schema | 121 KB | 3,251 | ~3 s (server cold-start) |

All three are well under the 1M context window. The differences are about
**richness, structure, and where the data lives** — not size.

---

## What's actually IN each, side by side: the `navigate` subcommand

### Recursive `--help` (the human-readable Rich-rendered form)

```
=== navigate ===

 Usage: selenium-browser navigate [OPTIONS] URL

 Navigate to a URL.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    url      TEXT  URL to navigate to. [required]                           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --fresh                             Close and reopen browser.                │
│ --browser      -b      TEXT         chrome or chromium.                      │
│ --har                               Enable HAR capture (requires --fresh).   │
│ --init-script          TEXT         JS to inject before page load            │
│                                     (repeatable).                            │
│ --format       -f      [text|json]  Output format. [default: text]           │
│ --help                              Show this message and exit.              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

**What's there:** flag names, short flags, help text, types as Rich-rendered
strings, required marker (`*`), defaults (`[default: text]`).
**What's NOT there:** structured types for parsers, default values for
non-string params, examples beyond the help string, programmatically-iterable
form. An agent has to regex-parse this if it wants structure.

### `Click.Context.to_info_dict()` (the introspection JSON)

```json
{
  "name": "navigate",
  "params": [
    {
      "name": "url",
      "param_type_name": "argument",
      "opts": ["url"],
      "type": {"param_type": "String", "name": "text"},
      "required": true, "nargs": 1, "default": null
    },
    {
      "name": "fresh",
      "param_type_name": "option",
      "opts": ["--fresh"],
      "type": {"param_type": "Bool", "name": "boolean"},
      "required": false, "default": false,
      "help": "Close and reopen browser.",
      "is_flag": true, "hidden": false
    },
    {
      "name": "browser",
      "param_type_name": "option",
      "opts": ["--browser", "-b"],
      "type": {"param_type": "String", "name": "text"},
      "required": false, "default": null,
      "help": "chrome or chromium."
    }
    /* ...4 more params... */
  ],
  "help": "Navigate to a URL.",
  "short_help": null,
  "hidden": false,
  "deprecated": false
}
```

**What's there:** every param structurally typed, all flags including short
forms, defaults as actual JSON values, hidden/deprecated flags, recursive into
sub-groups.
**What's NOT there:** rich descriptions (only the one-line `help=`), examples,
JSON Schema validators (e.g., `enum` for `[text|json]`), the multi-paragraph
docstrings the source function actually has.

### `fastmcp list --json` (the MCP wire form)

```json
{
  "name": "navigate",
  "description": "Load a URL and establish browser session. Entry point for all browser automation.\n\nAfter navigation completes, call get_aria_snapshot('body') to understand page structure\nbefore interacting with elements.\n\nArgs:\n    url: Full URL (http:// or https://)\n    fresh_browser: If True, creates clean session (no cache/cookies)\n    enable_har_capture: If True, enables performance logging for HAR export.\n    ...",
  "inputSchema": {
    "$defs": {
      "Browser": {"enum": ["chrome", "chromium"], "type": "string"}
    },
    "properties": {
      "url": {"title": "Url", "type": "string"},
      "fresh_browser": {"default": false, "type": "boolean"},
      "enable_har_capture": {"default": false, "type": "boolean"},
      "init_scripts": {
        "anyOf": [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}],
        "default": null
      },
      "browser": {
        "anyOf": [{"$ref": "#/$defs/Browser"}, {"type": "null"}],
        "default": null
      }
    },
    "required": ["url"]
  }
}
```

**What's there:** full multi-paragraph docstring as `description`, JSON Schema
typing with `$defs`/`enum`/`anyOf`/`$ref`, every nullable union typed correctly.
**What's NOT there:** CLI flag names (no `--fresh`, no `-b`), no concept of
short flags, no concept of rich-formatted help panels.

---

## Why all three exist

**They describe the same underlying functions but capture different layers of
the surface.** Each is the canonical answer for a specific consumer:

- The CLI exposes the function via Typer decorators → `--help` and
  `to_info_dict()` see *that* surface (kebab-case names, flag forms,
  CLI-rendered help)
- The MCP server exposes the function via FastMCP's Pydantic-typed
  registration → `fastmcp list` sees *that* surface (snake_case names,
  full docstrings, JSON Schema)
- They overlap because they wrap the same underlying logic but they are
  **not the same artifact**

---

## Click vs Typer (verified empirically)

> *"I would think that everything supported by Click is supported by Typer, but
> not the reverse."* — yes, with a refinement. **Typer is a superset of Click,
> built on Click.**

```
$ uv run python -c "import click, typer; \
  print('typer.Group bases:', typer.main.TyperGroup.__bases__); \
  print('typer.echo == click.echo?', typer.echo == click.echo)"

typer.Group bases: (<class 'click.core.Group'>,)
typer.echo == click.echo? True
```

Concretely:

- `typer.main.TyperGroup` directly subclasses `click.core.Group`
- `typer.echo` IS `click.echo` (identity, not just compatible)
- `typer.Argument` and `typer.Option` live in `typer.params` but produce
  `click.Argument` / `click.Option` instances internally
- Anything that consumes a Click `Group` (like `Context.to_info_dict()`) works
  on a Typer app via the one-line conversion `typer.main.get_command(app)`

What Typer adds on top of Click:

- Type annotations as parameter spec (`def navigate(url: str, fresh: bool =
  False)`) instead of decorators
- Auto-generated Rich panels (`rich_help_panel="Navigation"`)
- Auto shell-completion
- Cleaner DX for nested commands

None of these *break* Click's introspection — they all reduce to Click objects
underneath. **`to_info_dict()` is the supported, public Click API for walking
the tree, and it works on Typer apps unchanged.**

---

## CLI / MCP relationship in this workspace (the "veneer with deviations")

> *"The CLI is a veneer on top of the MCP, but there are things that deviate,
> like pipeline that are just more convenient in a CLI."*

This is exactly right. Empirically:

```
CLI subcommands (34):  capture-web-vitals, clear-proxy, click, configure-proxy,
                       download-resource, execute-javascript,
                       export-chrome-profile-state, export-har,
                       get-aria-snapshot, get-console-logs,
                       get-focusable-elements, get-interactive-elements,
                       get-page-html, get-page-text, get-resource-timings,
                       get-visual-tree, hover, install-completions,
                       list-chrome-profiles, navigate,
                       navigate-with-profile-state, pipeline, press-key,
                       resize-window, save-profile-state, screenshot, scroll,
                       set-blocked-urls, show-completion, sleep, type-text,
                       uninstall-completions, wait-for-network-idle,
                       wait-for-selector

MCP tools (30):        same 30 (snake-cased)

CLI-only (4):          install-completions, uninstall-completions,
                       show-completion, pipeline
```

The 4 CLI-only commands break down into:

- **Shell-tooling 3** (`install-completions`, `uninstall-completions`,
  `show-completion`) — install/print zsh-bash-fish tab completions for the
  binary. Has no MCP equivalent because MCP servers don't have a "binary in
  PATH" surface.
- **`pipeline`** (1) — batches multiple CLI calls into one process to amortize
  startup, share the same browser session, and short-circuit on first failure.
  Pure CLI ergonomics; no MCP equivalent because MCP calls are already
  individually addressable via the wire protocol with the same browser state.

**Implication: the MCP and CLI surfaces are not interchangeable.** The MCP is
the *core* surface; the CLI is the core surface plus shell-only ergonomics.
Anything we ship for "AI manifest" needs to surface the **CLI's** view if the
agent will be calling the CLI, because `pipeline` (and similar CLI-only
features) are otherwise invisible.

---

## Why ToolSearch doesn't replace this

> *"The MCP vector seems useful, but that seems solved already, right? The
> tool search built goes after that apparatus."*

ToolSearch (`anthropics/claude-code` in-binary feature) is an **agent-side
filter** over **already-loaded** MCP tools. It assumes:

1. The agent has the MCP server loaded into its tool registry
2. The user's agent harness (Claude Code, Cursor, etc.) implements ToolSearch

It answers the question: *"Among the 200 tools currently loaded, which ones
match `<query>`?"* — semantic search over an already-known surface.

It does NOT answer:

- *"What does this CLI offer if I'm running it from bash?"* — agent isn't in
  MCP context
- *"What does this CLI offer that ISN'T in the MCP server?"* (`pipeline`,
  `install-completions`) — those are invisible to ToolSearch by definition
- *"What CLIs and tools are available in this repo I just opened?"* — agent
  has nothing loaded yet
- *"What's the exact CLI flag form for invoking this from a shell script?"* —
  ToolSearch returns MCP tool names, not CLI invocation strings

The CLI-manifest problem is **upstream** of ToolSearch: it's how an agent
discovers a CLI's surface in the first place, before any MCP loading or
filtering happens. They're complementary, not redundant.

---

## Examples — where they live, what's preserved across surfaces

The user's question: *"the help on every subcommand, which includes all the
examples"* — let's see where examples actually exist.

### Source-of-truth: docstring on the MCP function

```python
async def navigate(
    url: str,
    fresh_browser: bool = False,
    ...
) -> NavigationResult:
    """Load a URL and establish browser session. Entry point for all
    browser automation.

    After navigation completes, call get_aria_snapshot('body') to understand
    page structure before interacting with elements.

    Example - API interception:
        navigate(
            "https://example.com",
            fresh_browser=True,
            init_scripts=['''
                window.__apiCapture = [];
                ...
            ''']
        )
    """
```

### Surfaced through each artifact

| Surface | Multi-paragraph description | Code examples | Args descriptions |
|---|---|---|---|
| `--help` | ❌ truncated to 1 line | ❌ stripped | ⚠️ from `help=` arg only |
| `to_info_dict()` | ❌ same as `--help` | ❌ stripped | ⚠️ from `help=` arg only |
| `fastmcp list --json` | ✅ full docstring | ✅ preserved verbatim | ✅ parsed from `Args:` block |

**The MCP surface is the only one that preserves examples.** This is because
FastMCP uses griffe to parse the function's docstring (multi-line description,
`Args:` block, examples) into the JSON Schema's `description` field, while
Click only knows about the one-line `help=` strings declared on the
`typer.Option(..., help="...")` calls.

This is a real asymmetry. If we want examples in the AI manifest, we need
either:

1. Push examples down into Typer's `help=` strings (verbose, duplicates the
   docstring)
2. Read the function's docstring at manifest-emit time and merge it into the
   `to_info_dict()` output (custom code, a few lines of inspection)
3. Use the MCP's `fastmcp list --json` output as the canonical AI manifest
   (free, but loses the 4 CLI-only commands)

---

## Tradeoff matrix

| Property | `--help` recursive dump | `to_info_dict()` JSON | `fastmcp list --json` |
|---|---|---|---|
| Effort to add | None (already exists, just iterate) | ~10 lines in `cli/main.py` | None (already exists in fastmcp) |
| Output format | Rich-rendered text | Click-shaped JSON | JSON Schema (industry-standard) |
| Captures CLI-only flags | ✅ | ✅ | ❌ |
| Captures CLI-only commands (`pipeline`) | ✅ | ✅ | ❌ |
| Captures multi-paragraph descriptions | ❌ | ❌ | ✅ |
| Captures code examples | ❌ | ❌ | ✅ |
| Programmatically parseable | ⚠️ regex | ✅ | ✅ |
| Cold-start cost | ~5s (forks per subcommand) | ~50ms | ~3s (MCP boot) |
| Requires running server | ❌ | ❌ | ✅ |
| Has shell flag form (`--fresh`, `-b`) | ✅ | ✅ | ❌ |
| Has JSON Schema types (`enum`, `anyOf`) | ❌ | ❌ | ✅ |

No surface is strictly best. **Each captures a real subset of the truth that
the others lose.**

---

## Recommendations

**The empirical result above suggests two complementary additions, not one
"winner":**

### Recommendation 1: `selenium-browser manifest` subcommand

A new top-level CLI subcommand (~30 lines, hidden from `--help` so we don't
clutter human surface) that emits a **merged JSON document** combining:

- `to_info_dict()` for the CLI shape (kebab names, flags, defaults,
  required, hidden) — including `pipeline` and shell-completion commands
- The function's **full docstring** read via `inspect.getdoc()` for each
  Typer-registered function — recovers the multi-paragraph descriptions and
  code examples that `to_info_dict()` drops

Output is selenium-browser-shaped JSON, not standard JSON Schema. Acceptable
because the consumer is an AI loading it into context, not a generic JSON
Schema validator.

```bash
selenium-browser manifest          # full JSON dump
selenium-browser manifest --summary # one-line-per-command table
```

### Recommendation 2: Document `fastmcp list` as the parallel AI surface for the MCP

When the agent already has the MCP server loaded (i.e., it's invoking
`mcp__selenium-browser__navigate` rather than the CLI), the MCP's
self-description via the standard `tools/list` call IS the AI manifest. No
work to do; just document it.

### What NOT to do

- Don't try to make one surface own everything — `pipeline` only makes sense
  on the CLI side; multi-paragraph docstrings only flow through the MCP side
- Don't invent a new schema (CLIbrary, AgentSpec, etc.) — fragmented, no winner
- Don't ship a SKILL.md until we have manifest output to point to

---

## Open questions (need a human decision)

1. **Should the docstring backfill happen at manifest emit time** (live
   inspection of source) **or be cached at build time** (committed JSON in
   repo)? Live = always accurate, no drift. Cached = no Python runtime needed
   to read, indexable by document-search.

2. **Manifest output JSON shape — flat or nested?** Flat (one top-level array
   of `{name, args, options, help, examples}`) is friendlier for LLMs to scan;
   nested (mirroring the Click tree) is more accurate when we add subcommand
   groups later.

3. **Where does `selenium-browser manifest --json` write to by default —
   stdout or a known file path?** Stdout is the unix way; a known path
   (`~/.claude-workspace/manifests/selenium-browser.json`) is friendlier for
   doc indexing.

4. **Workspace-wide convention?** Same `manifest` subcommand pattern for
   `claude-session`, `document-search`, `python-interpreter`, `claude-login`?
   Or selenium-browser only, see if it's useful, then propagate?

5. **Is `pipeline` still the only CLI-only "real" command, or are there others
   we should expect?** This affects whether the merge-with-fastmcp-list option
   could ever serve as a single source of truth.

---

## Artifacts

The three reference outputs generated for this paper live at
`/tmp/ai-cli-paper/` (until reboot):

- `recursive_help.txt` — 660 lines, 67 KB
- `to_info_dict.json` — 3,940 lines, 107 KB
- `fastmcp_list.json` — 3,251 lines, 121 KB

Re-generate with the commands shown at the top of each section.