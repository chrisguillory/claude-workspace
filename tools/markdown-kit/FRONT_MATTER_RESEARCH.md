# YAML Front Matter: Comprehensive Research Report

**Date:** March 2026
**Context:** markdown-kit.js currently passes YAML front matter through to `marked`, which renders the `---` delimiters as `<hr>` tags and shows YAML content as visible paragraph text. This report covers the full landscape and recommends a solution.

---

## 1. What IS YAML Front Matter?

### Origin and History

YAML front matter was invented by **Tom Preston-Werner** in **December 2008** as part of [Jekyll](https://jekyllrb.com/docs/front-matter/), the static site generator that powers GitHub Pages. The motivation was simple: blog posts need metadata (title, date, layout template) but Markdown has no native metadata syntax. Jekyll's solution: a block of YAML at the very top of the file, delimited by triple dashes.

```yaml
---
title: My Blog Post
date: 2026-03-26
author: Chris Guillory
tags: [home, construction]
---

The actual markdown content starts here.
```

This convention spread to every major static site generator (Hugo, Gatsby, Eleventy, Astro, Next.js, VitePress) and note-taking tool (Obsidian, Notion export, Logseq). It is now a **de facto standard** for embedding metadata in Markdown files, despite never being part of any Markdown specification (CommonMark, GFM, or original Markdown).

### The `---` Delimiter Problem

The `---` delimiter creates an inherent ambiguity. In standard Markdown, `---` on its own line is a **thematic break** (`<hr>`). YAML front matter repurposes this same syntax. The disambiguation rule is positional: front matter must be the **very first thing** in the file (line 1), with no preceding whitespace or blank lines. A `---` anywhere else in the document is a thematic break.

YAML also supports `...` as a closing delimiter (per the YAML spec), but `---` / `---` is the universal convention.

### Common Fields

| Field | Purpose | Used By |
|-------|---------|---------|
| `title` | Document title | Jekyll, Hugo, Obsidian, pandoc, Typora |
| `author` | Author name | pandoc, Typora (PDF metadata) |
| `date` | Creation/publish date | Jekyll, Hugo, Obsidian |
| `tags` | Categorization | Jekyll, Hugo, Obsidian |
| `description` | Summary/excerpt | Hugo, Astro, SEO |
| `layout` | Template selection | Jekyll, Hugo |
| `draft` | Publish status | Hugo, Astro |
| `subject` | PDF metadata | pandoc, Typora |
| `keywords` | PDF metadata / SEO | pandoc, Typora |

There is no formal schema -- any valid YAML key-value pair is allowed. Tools extract the fields they care about and ignore the rest.

---

## 2. How Different Tools Handle Front Matter

### Typora

**Editor display:** Typora renders front matter as a **collapsible gray metadata block** at the top of the document. It shows the raw YAML in a special code-like container, distinct from normal document content. Users can collapse or expand it.

**PDF export:** Typora **strips the front matter from the visible PDF content** but uses specific fields to set PDF document metadata:
- `title` -> PDF Title property
- `author` -> PDF Author property
- `creator` -> PDF Creator property
- `subject` -> PDF Subject property
- `keywords` -> PDF Keywords property (falls back to `tags` if `keywords` is absent)

**Configuration:** The preference "Read and overwrite export settings from YAML front matters" must be enabled for YAML-based export customization. Typora also supports per-document settings via front matter keys like `typora-root-url`, `typora-header`, `typora-footer`, and `typora-append-head`.

**Key insight:** Typora treats front matter as **metadata, not content**. It never renders the YAML text in the PDF body.

**Typora preferences:** No front-matter-specific keys exist in `defaults read abnerworks.Typora` on this machine. The behavior appears to be hardcoded: front matter is always parsed and stripped on export.

### Obsidian

**Editor display:** Since Obsidian v1.4 (2023), YAML front matter is displayed as **"Properties"** -- a structured key-value UI panel at the top of the note, not raw YAML text. Each property gets a typed input (text, date, checkbox, list, number). Users can toggle between the Properties UI view and raw YAML source view.

**Reading view:** Properties are **hidden by default** in reading mode. There is an editor setting to show/hide the properties section in reading view. Even when shown, they appear as the structured Properties panel, not raw YAML.

**Export:** Obsidian's native PDF export strips front matter from the output. When using the Obsidian-Pandoc plugin for export, front matter handling depends on pandoc's behavior (see below).

**Key insight:** Obsidian pioneered the idea that front matter is **structured metadata with a GUI**, not raw text to display.

### Hugo / Jekyll

**Behavior:** Front matter is **consumed and stripped**. The YAML data becomes template variables (`.Title`, `.Date`, `.Params.custom_field`) available to the site's layout templates. The rendered HTML page shows whatever the template decides -- typically the title becomes an `<h1>`, the date appears in a byline, etc. The raw YAML is never shown.

**Hugo additionally supports** TOML (`+++` delimiters) and JSON (`{}`/`{}` delimiters) front matter formats.

**Key insight:** SSGs treat front matter as **structured input to templates**. The template, not the front matter, controls what appears in the rendered output.

### GitHub

**Rendering:** GitHub renders YAML front matter as a **horizontal table** at the top of the markdown preview. Keys become column headers and values become the single data row.

**Known limitation:** This table format does not scale well. With many front matter fields, the table becomes very wide and hard to read. There is an [open issue](https://github.com/github/markup/issues/1490) requesting a vertical two-column layout instead.

**Key insight:** GitHub is the **only major tool** that renders front matter as visible content in the default view. It chose to show it because GitHub markdown files are often viewed without any separate metadata panel.

### VS Code

**Built-in preview:** VS Code's built-in Markdown preview has a setting `"markdown.previewFrontMatter"` with two values:
- `"hide"` (default) -- front matter is stripped from preview
- `"show"` -- front matter is shown as raw code block

**Extensions:**
- [Markdown YAML Preamble](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-yaml-preamble) -- renders front matter as a table (similar to GitHub)
- [Front Matter CMS](https://marketplace.visualstudio.com/items?itemName=eliostruyf.vscode-front-matter) -- full CMS-like metadata management UI

**Key insight:** VS Code defaults to **hiding** front matter but makes it configurable.

### Notion

**No native equivalent.** Notion uses **database properties** as its metadata system. These are structured fields (text, date, select, multi-select, person, etc.) on database pages. When exporting Notion pages to Markdown, tools like `notion-to-md` can optionally convert database properties to YAML front matter, but Notion itself has no YAML front matter concept.

**Key insight:** Notion's approach validates that **metadata should be structured, not raw text**.

### pandoc

**pandoc is the gold standard** for front matter handling. It:
1. Parses YAML front matter (also supports `---`/`...` closing delimiter)
2. Uses `title`, `author`, `date` to generate a formatted title block in the output
3. Injects metadata into PDF properties (via LaTeX)
4. Supports `--metadata` and `--metadata-file` CLI overrides
5. Allows front matter fields to be used as template variables

pandoc also supports **multiple YAML metadata blocks** in a single document (they merge), and metadata can include inline Markdown formatting.

**Key insight:** pandoc treats front matter as **both metadata AND potential content** -- it can populate PDF properties AND render a title block in the document body.

---

## 3. Rendering Options for markdown-kit.js

### Option A: Strip Silently (Typora/Hugo model)

**Behavior:** Parse front matter, remove it from the markdown before passing to `marked`. Use extracted fields for PDF metadata only.

**Pros:**
- Matches Typora behavior (our reference implementation)
- Clean PDF output -- no metadata clutter
- Front matter serves its intended purpose (metadata, not content)

**Cons:**
- Information loss -- reader of the PDF doesn't see the metadata
- User explicitly said "if I print a PDF and it doesn't have the front matter, that seems like I'm missing useful information"

### Option B: Render as Styled Metadata Block

**Behavior:** Parse front matter, render selected fields (title, author, date, etc.) as a styled header block at the top of the PDF.

**Example output:**
```
+--------------------------------------------------+
|  Title: Kitchen Appliance Package Summary         |
|  Author: Chris Guillory                           |
|  Date: March 26, 2026                             |
+--------------------------------------------------+
```

**Pros:**
- Important metadata is visible in the PDF
- Looks professional and intentional
- Selective -- only renders human-relevant fields, not technical ones

**Cons:**
- Opinionated about which fields to show
- May duplicate content if the document already has an H1 title
- Styling needs to look good across themes

### Option C: Render as Code Block

**Behavior:** Parse front matter, render the raw YAML in a styled `<pre>` block.

**Pros:**
- No information loss
- Simple implementation

**Cons:**
- Looks like debugging output, not a polished document
- Technical readers only

### Option D: Use for PDF Metadata Only + Optional Header (Recommended)

**Behavior:** Parse front matter, use fields to set PDF document metadata (title, author, subject, keywords). Optionally render a document header from selected fields. Strip the raw YAML from the body.

**Pros:**
- PDF metadata is properly set (searchable, accessible, professional)
- Clean document body by default
- Optional visible header for users who want it
- Follows pandoc's model (the most sophisticated approach)

**Cons:**
- More complex implementation (but manageable)

### Option E: Pass Through Raw (Current Broken Behavior)

**Behavior:** Don't parse front matter at all. Let `marked` handle it.

**Result:** `---` renders as `<hr>`, YAML content renders as paragraph text. This is incorrect -- no tool does this intentionally.

---

## 4. Best Practices for a PDF Generation Tool

### PDF Metadata (Always Do This)

Every PDF generation tool should set PDF document metadata from front matter fields. This is not optional -- it is a basic quality requirement:

- **Title**: From `title` field, or fall back to first H1, or fall back to filename
- **Author**: From `author` field
- **Subject**: From `subject` or `description` field
- **Keywords**: From `keywords` field, or fall back to `tags`
- **Creator**: Set to `markdown-kit.js` (identifies the tool)

PDF metadata matters for:
- **Search indexing** -- PDF search engines use these fields
- **Accessibility** -- Screen readers announce the title
- **File managers** -- macOS Finder, Windows Explorer show PDF metadata
- **Document management systems** -- Use metadata for cataloging

**Current state:** Chromium already sets the PDF `/Title` from the HTML `<title>` tag automatically (verified: our Kitchen Appliance Package PDF has Title = "Our Kitchen Appliance Package" from the H1 extraction). Author, Subject, and Keywords are not set.

**Enhancement needed:** For Author/Subject/Keywords, we would need to post-process the PDF with a library like `pdf-lib` since Puppeteer/Chromium does not expose these fields. However, the title extraction via `<title>` tag already works.

### Front Matter Visibility (Configurable)

The best approach is pandoc's: **strip by default, render optionally**.

- Default: front matter is parsed, stripped from body, used for PDF metadata
- Flag: `--front-matter render` to show a styled metadata header
- Flag: `--front-matter raw` to show the YAML as a code block

### Front Matter as CLI Override

Front matter fields should be able to override CLI options. This is powerful for per-document configuration:

```yaml
---
title: My Document
theme: github
width: 1200
pageless: false
---
```

This mirrors Typora's `typora-root-url` and `typora-header` per-document settings, and pandoc's metadata variable system.

---

## 5. Technical Implementation

### Library Choice: `gray-matter` vs `front-matter`

| Aspect | gray-matter | front-matter |
|--------|------------|--------------|
| Monthly downloads | 19.4M | 27.3M |
| Return object | `{ data, content, excerpt, matter, isEmpty }` | `{ attributes, body, bodyBegin, frontmatter }` |
| `test()` method | Yes -- `matter.test(string)` | Yes -- `fm.test(string)` |
| Stringify back | Yes -- `matter.stringify()` | No |
| Multiple formats | YAML, JSON, TOML, CoffeeScript | YAML only |
| Custom delimiters | Yes | No |
| Excerpt extraction | Yes (built-in) | No |
| Used by | Gatsby, Next.js, VitePress, Astro, Metalsmith | Express, HarpJS |
| Dependencies | `js-yaml`, `strip-bom-string`, `section-matter` | `js-yaml` |
| Last major update | 2022 (stable, mature) | 2020 (stable, mature) |

**Recommendation: `gray-matter`**

Despite slightly fewer downloads, gray-matter is the better choice because:
1. Used by the SSG ecosystem we're closest to (static document generation)
2. `matter.test()` for detection without full parsing
3. Excerpt support (potential future feature)
4. Returns both parsed `data` and raw `matter` (useful for `--front-matter raw` mode)
5. Has `isEmpty` flag for empty front matter blocks

### Integration with marked

The integration is straightforward -- gray-matter preprocessing happens **before** marked parsing:

```javascript
const matter = require('gray-matter');

// Parse front matter BEFORE marked
const { data: frontMatter, content: mdContent } = matter(mdSource);

// Pass only the content (front matter stripped) to marked
const htmlBody = marked.parse(mdContent);
```

This is the pattern recommended by the marked.js maintainers themselves (see [issue #485](https://github.com/markedjs/marked/issues/485)): front matter extraction is a preprocessing step, not a parser extension.

### Dependency Addition

Add to the `DEPS` object in markdown-kit.js:

```javascript
'gray-matter': '^4',
```

gray-matter has minimal dependencies (js-yaml, strip-bom-string, section-matter) and is well-suited for the persistent cache pattern.

### Implementation Sketch

```javascript
// ── Argument parsing additions ──────────────────────────────────────────
// --front-matter <mode>   strip (default) | render | raw
const frontMatterMode = opts['front-matter'] || 'strip';

// ── Front matter parsing (before marked) ────────────────────────────────
const grayMatter = require('gray-matter');
const parsed = grayMatter(mdSource);
const frontMatter = parsed.data;       // { title, author, date, ... }
const mdContent = parsed.content;       // markdown without front matter

// ── Use front matter for document title ─────────────────────────────────
// Priority: front matter title > first H1 > filename
const docTitle = frontMatter.title
  || (mdContent.match(/^#\s+(.+)$/m)?.[1]?.trim())
  || basename(inputPath, extname(inputPath));

// ── Use front matter for CLI overrides ──────────────────────────────────
// Front matter values are defaults; CLI flags take priority
const effectiveTheme = opts.theme !== 'pixyll' ? opts.theme
  : (frontMatter.theme || 'pixyll');
const effectiveWidth = opts.width !== '1400' ? parseInt(opts.width)
  : (frontMatter.width || 1400);

// ── Optional: render front matter as styled header ──────────────────────
let frontMatterHtml = '';
if (frontMatterMode === 'render' && !parsed.isEmpty) {
  const fields = [];
  if (frontMatter.title) fields.push(`<dt>Title</dt><dd>${esc(frontMatter.title)}</dd>`);
  if (frontMatter.author) fields.push(`<dt>Author</dt><dd>${esc(frontMatter.author)}</dd>`);
  if (frontMatter.date) fields.push(`<dt>Date</dt><dd>${esc(String(frontMatter.date))}</dd>`);
  // ... other fields
  if (fields.length > 0) {
    frontMatterHtml = `<dl class="front-matter-block">${fields.join('')}</dl>`;
  }
} else if (frontMatterMode === 'raw' && parsed.matter) {
  frontMatterHtml = `<pre class="front-matter-raw"><code>${esc(parsed.matter)}</code></pre>`;
}

// ── Parse markdown ──────────────────────────────────────────────────────
const htmlBody = marked.parse(mdContent);  // front matter already stripped
const resolvedBody = frontMatterHtml + htmlBody;  // prepend if rendered
```

### CSS for Rendered Front Matter

```css
/* Styled metadata block (--front-matter render) */
.front-matter-block {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 12px 20px;
  margin: 0 0 24px 0;
  background: #f8f9fa;
  font-size: 0.9em;
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 16px;
}
.front-matter-block dt {
  font-weight: 600;
  color: #555;
}
.front-matter-block dd {
  margin: 0;
}

/* Raw YAML block (--front-matter raw) */
.front-matter-raw {
  border: 1px solid #e1e4e8;
  border-radius: 4px;
  padding: 12px 16px;
  margin: 0 0 24px 0;
  background: #f6f8fa;
  font-size: 0.85em;
}
```

---

## 6. Typora-Specific Behavior Summary

| Aspect | Typora Behavior | Our Current Behavior | Recommended |
|--------|----------------|---------------------|-------------|
| Front matter in editor | Collapsible gray metadata block | N/A (CLI tool) | N/A |
| Front matter in PDF body | **Stripped** (not visible) | **Broken** (rendered as hr + text) | Strip by default |
| PDF Title metadata | Set from `title` field | Set from H1 or filename via `<title>` tag | Set from front matter `title` (priority) |
| PDF Author metadata | Set from `author` field | **Not set** | Set from front matter `author` |
| PDF Keywords metadata | Set from `keywords` or `tags` | **Not set** | Set from front matter (future: pdf-lib) |
| Per-doc settings via YAML | `typora-root-url`, `typora-header`, etc. | None | Support `theme`, `width`, `pageless` |
| Pref to control behavior | "Read and overwrite export settings from YAML front matters" | None needed (CLI flag) | `--front-matter` flag |

---

## 7. Recommendation

### Proposed `--front-matter` Flag

```
--front-matter <mode>    How to handle YAML front matter (default: strip)

Modes:
  strip     Parse and remove from output. Use for PDF metadata only. (default)
  render    Show title/author/date as a styled header block in the PDF.
  raw       Show the raw YAML as a code block at the top of the PDF.
```

### Implementation Priority

**Phase 1 (do now):**
1. Add `gray-matter` to dependencies
2. Parse front matter before passing to `marked` (fixes the broken hr + text rendering)
3. Use `title` from front matter as priority source for `<title>` tag (already flows to PDF Title via Chromium)
4. Strip front matter from body by default
5. Add `--front-matter` flag with `strip` (default), `render`, and `raw` modes

**Phase 2 (later):**
6. Support front matter fields as CLI overrides (`theme`, `width`, `pageless`)
7. Add CSS for `render` mode to both pixyll and github themes
8. Add `pdf-lib` post-processing for Author/Subject/Keywords PDF metadata

### Why This Approach

1. **Fixes the immediate bug** -- front matter no longer renders as broken HTML
2. **Matches Typora** -- strip is the default, matching our reference implementation
3. **Respects the user's insight** -- `--front-matter render` gives visibility when wanted
4. **No information loss** -- `--front-matter raw` shows everything
5. **Future-proof** -- front matter as CLI override opens powerful per-document configuration
6. **Minimal complexity** -- gray-matter is one dependency, preprocessing is ~10 lines

---

## Sources

- [Jekyll Front Matter documentation](https://jekyllrb.com/docs/front-matter/)
- [Typora YAML Front Matter support](https://support.typora.io/YAML/)
- [Typora Export documentation](https://support.typora.io/Export/)
- [marked.js issue #485 -- front matter support](https://github.com/markedjs/marked/issues/485)
- [gray-matter on GitHub](https://github.com/jonschlinkert/gray-matter)
- [front-matter on GitHub](https://github.com/jxson/front-matter)
- [GitHub YAML front matter table rendering issue](https://github.com/github/markup/issues/1490)
- [VS Code Markdown YAML Preamble extension](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-yaml-preamble)
- [Hugo Front Matter documentation](https://gohugo.io/content-management/front-matter/)
- [pandoc Metadata Blocks documentation](https://pandoc.org/demo/example33/8.10-metadata-blocks.html)
- [Puppeteer PDF metadata issue #3054](https://github.com/puppeteer/puppeteer/issues/3054)
- [Obsidian Properties documentation](https://help.obsidian.md/properties)
