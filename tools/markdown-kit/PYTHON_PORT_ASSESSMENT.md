# Should markdown-kit Be Ported from JavaScript to Python?

An honest, comprehensive assessment. March 2026.

---

## 1. What the Tool Actually Does

markdown-kit.js is a 1,444-line Node.js ESM script that converts Markdown to PDF with Typora-compatible rendering. It is surprisingly complex. The major components:

1. **Dependency bootstrapping** -- Self-installing npm packages into `~/.cache/markdown-kit` using `createRequire()` to work around Node's broken `npx -p` module resolution.

2. **Markdown parsing** -- GFM via `marked` with 10 extensions: syntax highlighting (highlight.js), KaTeX math, footnotes, GFM alerts, heading IDs, emoji shortcodes, plus three custom inline extensions (==highlight==, ~subscript~, ^superscript^).

3. **Mermaid diagrams** -- Injecting the mermaid.js browser bundle into Puppeteer and rendering code blocks as SVG in-page.

4. **Typora-compatible rendering** -- Theme CSS loading, custom highlight.js theme, KaTeX font resolution, image path resolution (file:// vs HTTP), front matter handling, TOC generation.

5. **PDF generation** -- Two engines: Chromium (Puppeteer `page.pdf()`) and WebKit (native macOS Swift binary). Two-pass pageless mode: measure content height, then generate exact-fit single-page PDF.

6. **PDF post-processing** -- Three operations via separate tools:
   - Internal link annotations via compiled Swift tool (`add-links`) using Apple PDFKit
   - Document outline/bookmarks via pdf-lib (JavaScript)
   - PDF metadata injection via pdf-lib

7. **Live preview server** -- HTTP server with SSE-based auto-reload, file watching, static asset serving for images/fonts/themes.

8. **Platform-specific integration** -- Swift companion binaries (`add-links`, `webkit-pdf`, `verify-links`) using Apple PDFKit/WKWebView. macOS-only features.

---

## 2. Dependency-by-Dependency Comparison

### Browser Automation / PDF Engine

| JS | Python | Verdict |
|---|---|---|
| **puppeteer** ^24 (headless Chromium, `page.pdf()`) | **playwright** (Python) | Playwright Python's `page.pdf()` is **Chromium-only** -- no WebKit PDF mode. Functionally equivalent to Puppeteer for Chromium PDF generation. API is nearly identical. |

**Critical finding**: Playwright Python does NOT support WebKit PDF generation. You get the error "Page.pdf: PDF generation is only supported for Headless Chromium." The JS tool's WebKit path shells out to a compiled Swift binary anyway, so this is a wash -- both languages would need the Swift tool for WebKit.

### Markdown Parsing

| JS | What it does | Python equivalent | Quality comparison |
|---|---|---|---|
| **marked** ^15 | GFM Markdown -> HTML | **markdown-it-py** | markdown-it-py is a faithful Python port of markdown-it (the JS library). 100% CommonMark compliant. Plugin ecosystem is smaller than marked's but growing. The port is maintained by the Executable Books Project (Jupyter ecosystem). **Good equivalent.** |
| | | **Python-Markdown** | Older, established, huge extension ecosystem (pymdown-extensions alone has 20+ extensions). Not CommonMark compliant. Slower than markdown-it-py. Used by MkDocs/Material. |
| | | **mistune** ^3 | Fastest Python parser. Plugin system exists but extension ecosystem is thin. Not fully CommonMark compliant. |

**Recommendation**: markdown-it-py is the closest equivalent to marked. Python-Markdown + pymdown-extensions is the most battle-tested for feature coverage.

### Syntax Highlighting

| JS | Python | Verdict |
|---|---|---|
| **marked-highlight** ^2 + **highlight.js** ^11 | **Pygments** | Pygments is **superior** to highlight.js. 598 lexers vs ~190 languages. Richer tokenization (more token types = more colorable spans). Server-side rendering. GitHub uses Pygments. The custom hljs-typora.css theme would need to be converted to a Pygments style, which is straightforward. **Win for Python.** |

### Math Rendering

| JS | Python | Verdict |
|---|---|---|
| **marked-katex-extension** ^5 + **katex** ^0.16 | **No pure-Python KaTeX.** Options: (1) mdit-py-plugins has `dollarmath` for parsing `$...$` and `$$...$$`, but rendering to HTML requires shelling out to KaTeX CLI or using MathJax. (2) pymdown-extensions has Arithmatex which preserves math for client-side rendering. (3) Pre-render with `katex` npm CLI. | **Significant gap.** The JS tool renders KaTeX entirely in-process (the katex npm package is a pure JS library that emits HTML+CSS). Python has no equivalent pure-Python math renderer. You'd need to either: (a) keep Node.js as a dependency just for KaTeX, (b) shell out to the `katex` CLI, or (c) use MathJax (heavier, different output). This is the single biggest obstacle to a clean port. |

### Mermaid Diagrams

| JS | Python | Verdict |
|---|---|---|
| **mermaid** ^11 (browser bundle injected into Puppeteer page) | **mermaid-cli-python** (uses Playwright Chromium), **mmdc** (uses PhantomJS via phasma), **mermaid-py** (uses mermaid.ink online service) | The JS tool injects mermaid.js directly into the Puppeteer page context -- elegant, zero extra processes. In Python, you'd either: (a) do the same thing with Playwright (inject mermaid.min.js into the page -- this works identically), (b) pre-render to SVG via a separate tool, or (c) use an online API. Option (a) is the right approach and works fine. **Equivalent.** |

### Heading IDs & TOC

| JS | Python | Verdict |
|---|---|---|
| **marked-gfm-heading-id** ^4 | mdit-py-plugins `anchors_plugin`, or Python-Markdown's `toc` extension | Both Python parsers generate heading IDs and can produce TOC. **Equivalent.** |

### Emoji

| JS | Python | Verdict |
|---|---|---|
| **marked-emoji** ^2 + **node-emoji** ^2 | **emoji** (pip), pymdown-extensions `emoji` extension | Python `emoji` library supports shortcodes. pymdown-extensions provides rich emoji integration with Gemoji/Twemoji indexes. **Equivalent.** |

### Footnotes

| JS | Python | Verdict |
|---|---|---|
| **marked-footnote** ^1 | mdit-py-plugins `footnote_plugin`, Python-Markdown `footnotes` extension | Both are well-maintained. **Equivalent.** |

### GFM Alerts

| JS | Python | Verdict |
|---|---|---|
| **marked-alert** ^2 | **markdown-gfm-admonition** (PyPI), pymdown-extensions admonition + quotes callout mode | `markdown-gfm-admonition` specifically supports GitHub's `> [!NOTE]` syntax. pymdown-extensions also added this. **Equivalent.** |

### Front Matter

| JS | Python | Verdict |
|---|---|---|
| **gray-matter** ^4 | **python-frontmatter** | Direct equivalent. Same API pattern (parse, get data + content). **Equivalent.** |

### PDF Manipulation

| JS | Python | Verdict |
|---|---|---|
| **pdf-lib** ^1 (bookmarks, metadata) | **PyMuPDF** (fitz), **pypdf**, **pikepdf** | PyMuPDF is **dramatically superior** to pdf-lib. Full outline/bookmark tree manipulation, annotation support (including link annotations with proper /Dest arrays), metadata injection, and much more. PyMuPDF could potentially replace both pdf-lib AND the Swift add-links tool, since it can create link annotations with proper destination arrays that macOS Preview understands. **Major win for Python.** |

### Custom Inline Extensions

| JS | What | Python |
|---|---|---|
| Custom `marked` extension | `==highlight==` -> `<mark>` | pymdown-extensions `mark` extension, or mdit-py-plugins |
| Custom `marked` extension | `~subscript~` -> `<sub>` | pymdown-extensions `caret` + `tilde`, or custom plugin |
| Custom `marked` extension | `^superscript^` -> `<sup>` | pymdown-extensions `caret`, or custom plugin |

pymdown-extensions provides all three out of the box. **Equivalent or better** (no custom extension code needed).

---

## 3. The KaTeX Problem

This is the deal-breaker analysis. The JS tool renders math like this:

```
marked.use(markedKatexFn({ throwOnError: false }));
```

That single line gives you full KaTeX rendering because the `katex` npm package is a pure JavaScript library that runs in Node.js and emits complete HTML+CSS (no browser needed). The KaTeX CSS and fonts are loaded from the cached node_modules.

In Python, there is no equivalent pure-Python math renderer. Your options:

**Option 1: Shell out to `npx katex`**
- Requires Node.js installed. Defeats the purpose of porting.

**Option 2: Use MathJax (client-side)**
- Parse `$...$` with Arithmatex/dollarmath, emit MathJax script tags, let the browser render during PDF generation.
- Works but: heavier than KaTeX, different rendering output, requires network access or bundled MathJax.

**Option 3: Use KaTeX client-side**
- Parse math delimiters, emit raw LaTeX, inject KaTeX JS+CSS into the HTML, let the browser (Playwright Chromium) render it.
- This is what the JS tool effectively does for the PDF path anyway (KaTeX emits HTML that the browser renders to PDF).
- In Python: extract math blocks, wrap in KaTeX-compatible spans, inject KaTeX bundle.

**Option 4: Use `latex2mathml` (pure Python)**
- Converts LaTeX to MathML. Browsers render MathML natively (Chromium support landed in 2023).
- Quality is lower than KaTeX for complex expressions.

**Honest assessment**: Option 3 is viable and architecturally clean. You'd parse math in Python, emit LaTeX strings wrapped in the right HTML, and inject KaTeX JS+CSS into the page for browser-side rendering. The visual output would be identical because KaTeX is doing the rendering either way -- the question is just whether it runs in Node.js or in the Chromium page context. For a tool that already launches Chromium, this is fine.

---

## 4. The Swift Companion Tools

The tool has three compiled Swift binaries:

| Binary | Purpose | Python replacement? |
|---|---|---|
| `add-links` | Add internal link annotations using Apple PDFKit (macOS Preview compatibility) | **PyMuPDF** can create link annotations with proper /Dest entries. Would need testing to confirm macOS Preview compatibility. If it works, this eliminates the Swift dependency. If not, keep the Swift tool. |
| `webkit-pdf` | Generate PDF via macOS WKWebView (same engine as Typora) | No Python equivalent. This is an Apple-only API. Would remain as a Swift binary regardless of language choice. |
| `verify-links` | Verify internal links in generated PDFs | **PyMuPDF** can read and verify annotations. Trivial to port. |

**Key insight**: The WebKit engine path (`--engine webkit`) requires a native macOS binary regardless. A Python port would still need this Swift tool. The `add-links` tool might be replaceable by PyMuPDF, but would require careful testing with macOS Preview.app.

---

## 5. Your Ecosystem

Your `claude-workspace` is a Python-first environment:
- **5 MCP servers**, all Python (selenium-browser-automation, claude-session, document-search, python-interpreter, browser-automation)
- **uv workspace** with pyproject.toml, strict mypy, ruff, pytest
- Python 3.13, strict typing, comprehensive linting
- Dependency management via `uv` (fast, reliable, lockfiles)

The JS tool is an outlier in this ecosystem. It uses a bespoke dependency caching pattern (`~/.cache/markdown-kit` + `createRequire()`) because Node's module resolution fundamentally doesn't support the `uvx`-style "run from anywhere" pattern. This was well-documented in the tool itself:

> NOTE: npx -p does NOT work for this -- Node's module resolution (both ESM and CJS) resolves packages relative to the script's location, not npx's temp directory. This is fundamental to Node, not version-specific.

In Python, `uv tool install` or `uvx` solves this cleanly. A Python version would have zero bootstrapping complexity.

---

## 6. Option Assessment

### Option A: Keep JavaScript, Python MCP wrapper

**Architecture**: Python MCP server that shells out to `node markdown-kit.js`.

**Pros**:
- Zero risk. The JS tool works perfectly.
- MCP server is a thin `subprocess.run()` wrapper.
- Fastest to implement (hours, not days).

**Cons**:
- Requires Node.js installed on any machine running the MCP server.
- Two dependency ecosystems (npm cache + Python venv).
- Can't extend the tool in Python (want to add a feature? Write JS).
- The MCP server would be a second-class citizen -- it can't inspect intermediate results, modify the pipeline, or integrate deeply.

**Verdict**: Pragmatic but unsatisfying. Creates a permanent Node.js dependency in a Python ecosystem.

### Option B: Full port to Python

**Architecture**: Pure Python tool using Playwright + markdown-it-py (or Python-Markdown) + Pygments + PyMuPDF.

**Effort estimate**: 3-5 focused sessions. The tool is 1,444 lines but much of it is boilerplate (arg parsing, CSS loading, HTML templating) that ports trivially. The complex parts:

| Component | Effort | Risk |
|---|---|---|
| Arg parsing, file I/O, HTML assembly | Low | None |
| Markdown parsing + extensions | Medium | Need to verify all 10 extensions produce equivalent HTML |
| Syntax highlighting (Pygments) | Low | Better than current. Need to convert hljs-typora.css to Pygments style. |
| KaTeX math rendering | Medium | Client-side rendering in Chromium page (see Section 3) |
| Mermaid diagrams | Low | Same approach: inject mermaid.min.js into Playwright page |
| PDF generation (Chromium) | Low | `page.pdf()` API is nearly identical in Playwright |
| PDF post-processing (outlines, metadata) | Low | PyMuPDF is more capable than pdf-lib |
| Internal link annotations | Medium | PyMuPDF replaces Swift tool? Needs Preview.app testing |
| Live preview server | Low | Python's `http.server` + watchdog/watchfiles |
| Dependency management | **Eliminated** | `uv` handles everything. No bootstrap code needed. |

**What you gain**:
- Single language ecosystem. `uv tool install markdown-kit` just works.
- PyMuPDF is dramatically more capable than pdf-lib -- could enable new PDF features.
- Pygments produces richer syntax highlighting than highlight.js.
- Natural MCP server integration (import the library directly, no subprocess).
- Easier to extend for anyone in your ecosystem (you, Claude, collaborators).
- Proper packaging with pyproject.toml, type hints, tests.

**What you lose**:
- The `marked` extension ecosystem is larger and more actively maintained than any single Python parser's ecosystem. You'd need to verify every edge case.
- KaTeX server-side rendering is gone. Client-side rendering in the browser is equivalent for PDF output, but the HTML preview server would need KaTeX JS injected (same as current serve mode).
- Risk of rendering differences. The current tool is audited against Typora. A port would need re-auditing.
- Development time. 3-5 sessions is real time.

**Biggest risk**: Rendering parity. The current tool has been carefully tested against Typora's output. A Python port would need a full re-audit. Markdown parsers have subtle differences in edge cases (nested lists, inline formatting interactions, GFM table alignment). The rendering-test.md file and comparison screenshots exist specifically because of this.

**Verdict**: The strongest long-term option. Real effort, real risk, real payoff.

### Option C: Hybrid (Python orchestrator + JS for markdown parsing)

**Architecture**: Python tool that calls `node -e "..."` for markdown parsing, then does everything else in Python.

**Pros**: None worth mentioning.

**Cons**:
- Worst of both worlds. Two runtimes, complex IPC, fragile.
- Harder to maintain than either pure option.
- Still requires Node.js.

**Verdict**: No. This is engineering malpractice.

### Option D: Keep JavaScript for everything (including MCP server)

**Architecture**: JavaScript MCP server, JavaScript tool. No Python.

**Pros**:
- Single language (for this tool).
- No porting effort.

**Cons**:
- MCP server in JS is an outlier in your Python ecosystem.
- JS MCP ecosystem is less mature than Python's (fastmcp, etc.).
- You'd be maintaining JS code you'd rather write in Python.
- The dependency bootstrapping hack remains.

**Verdict**: If you never plan to build an MCP server for this, this is the lowest-effort path. But it's a dead end for integration.

---

## 7. What Do Popular Projects Use?

| Project | Language | Markdown Parser | PDF Engine |
|---|---|---|---|
| **MkDocs + Material** | Python | Python-Markdown | WeasyPrint (via mkdocs-pdf-export) or browser-based |
| **Pandoc** | Haskell | Custom parser | LaTeX, WeasyPrint, wkhtmltopdf, or Chromium |
| **Jupyter Book** | Python | MyST (markdown-it-py) | LaTeX |
| **Sphinx** | Python | reStructuredText / MyST | LaTeX |
| **mdBook** | Rust | pulldown-cmark | Chromium (via mdbook-pdf) |
| **Slidev** | JavaScript | markdown-it | Playwright Chromium |
| **Marp** | JavaScript | markdown-it | Playwright Chromium |
| **Docusaurus** | JavaScript | MDX | N/A (web only) |
| **Typora** | JavaScript (Electron) | Custom parser | Chromium (built-in) |

**Pattern**: Documentation tools are split roughly 50/50 between Python and JS. For Markdown-to-PDF specifically, Chromium-based rendering dominates. WeasyPrint is used for simpler documents (invoices, reports) but lacks JavaScript support needed for Mermaid/KaTeX.

Your tool's architecture (Markdown -> HTML -> Chromium -> PDF -> post-process) matches the modern pattern used by Slidev, Marp, and Typora itself. This architecture works equally well in Python (via Playwright) as in JS (via Puppeteer).

---

## 8. Recommendation

**Port to Python (Option B), but do it incrementally.**

Here is the reasoning:

### Why port:

1. **Ecosystem alignment.** Your entire tooling ecosystem is Python. This tool is the only Node.js dependency. Every time you extend it, you context-switch to a language you're less invested in.

2. **PyMuPDF is a genuine upgrade.** The current pdf-lib + Swift tool combination for bookmarks and link annotations is a workaround for pdf-lib's limitations. PyMuPDF handles all of this natively and may eliminate the need for the `add-links` Swift binary entirely.

3. **Pygments is a genuine upgrade.** Richer tokenization, more languages, battle-tested (GitHub uses it).

4. **Dependency management is a genuine upgrade.** The `createRequire()` bootstrap hack is clever but fragile. `uv tool install` is how this should work.

5. **MCP integration becomes native.** Instead of `subprocess.run(["node", "markdown-kit.js", ...])`, the MCP server imports the library directly. It can access intermediate results, modify the pipeline, stream progress.

### Why it's not a slam dunk:

1. **KaTeX has no Python equivalent.** Client-side rendering in the browser works, but it's a design change from server-side rendering. For the PDF path this is invisible (the browser renders it either way). For the HTML preview server, KaTeX JS must be injected (which the current serve mode already does).

2. **Rendering parity risk.** The tool has been carefully audited against Typora. A different markdown parser will produce subtly different HTML in edge cases. This requires re-auditing with the rendering-test.md document. Budget a full session just for this.

3. **The WebKit engine path stays in Swift.** The `webkit-pdf` binary is Apple-only and has no Python equivalent. This is fine -- it's an optional engine -- but it means the tool isn't pure Python regardless.

### Suggested approach:

**Phase 1**: Build the Python version as a new file alongside the JS version. Port the core pipeline: markdown parsing, HTML assembly, Chromium PDF generation. Use the rendering-test.md document to verify output parity. Keep the JS version working.

**Phase 2**: Port PDF post-processing. Test PyMuPDF for bookmarks, metadata, and internal link annotations. If PyMuPDF link annotations work in macOS Preview, retire the Swift `add-links` tool. If not, keep shelling out to it.

**Phase 3**: Add the live preview server. Port the SSE-based auto-reload.

**Phase 4**: Build the MCP server as a native Python integration.

**Phase 5**: Retire the JS version once the Python version is at feature parity and passes the rendering audit.

### Timeline estimate:

- Phase 1: 2-3 sessions
- Phase 2: 1 session
- Phase 3: 1 session
- Phase 4: 1 session
- Phase 5: 1 session (audit only)

Total: 6-8 sessions, spread over time. No urgency -- the JS version works fine.

---

## 9. Summary Table

| Factor | JS (current) | Python (port) | Winner |
|---|---|---|---|
| Markdown parsing ecosystem | marked (large, active) | markdown-it-py or Python-Markdown (adequate) | JS (slightly) |
| Syntax highlighting | highlight.js (good) | Pygments (better) | **Python** |
| Math rendering | KaTeX in-process (elegant) | KaTeX client-side in browser (fine for PDF) | JS (slightly) |
| Mermaid rendering | mermaid.js in Puppeteer | mermaid.js in Playwright | Tie |
| PDF post-processing | pdf-lib (limited) + Swift tool | PyMuPDF (comprehensive) | **Python** |
| Dependency management | createRequire() hack | uv (clean) | **Python** |
| Ecosystem fit | Outlier | Native | **Python** |
| MCP integration | subprocess wrapper | Native import | **Python** |
| Rendering parity risk | Proven, audited | Needs re-audit | JS |
| Effort to maintain status quo | Zero | 6-8 sessions to port | JS |

**Bottom line**: The port is worth doing. The JS tool works, but it's stranded in the wrong ecosystem. Python gives you better PDF manipulation, better syntax highlighting, cleaner dependency management, and native MCP integration. The main risk (rendering parity) is manageable with the existing test infrastructure. The main cost (KaTeX) has a clean workaround (browser-side rendering). Do it incrementally, keep the JS version as a reference, and retire it once parity is confirmed.
