#!/usr/bin/env node
//
// markdown-kit.js — Markdown to single-page PDF (no page breaks, auto-fit height)
//
// Usage:
//   node markdown-kit.js input.md [--stylesheet path/to/style.css] [--width 860]
//   node markdown-kit.js input.md --engine webkit   (macOS WKWebView, same as Typora)
//   node markdown-kit.js input.md --front-matter render   (show YAML front matter as header)
//   node markdown-kit.js input.md --no-pageless           (A4 pages with page numbers)
//   node markdown-kit.js input.md --no-pageless --footer "Draft — {{pageNumber}}/{{totalPages}}"
//   node markdown-kit.js input.md --serve                 (live preview with auto-reload)
//   node markdown-kit.js input.md --serve --host 0.0.0.0  (network access)
//
// Dependencies (puppeteer, marked) are auto-installed into a persistent cache
// on first run (~/.claude-workspace/tools/markdown-kit). No project-level node_modules needed.
// NOTE: npx -p does NOT work for this — Node's module resolution (both ESM
// and CJS) resolves packages relative to the script's location, not npx's temp
// directory. This is fundamental to Node, not version-specific. The persistent
// cache + createRequire() pattern is the standard workaround.
//
// Output: input.pdf alongside the input file.
//
// ── Typora Preference Alignment ─────────────────────────────────────────────
//
// Audit against Typora 1.12.6 preferences on this machine (March 2026).
// Preferences read from ~/Library/Preferences/abnerworks.Typora.plist.
//
// MATCHED — Rendering-affecting preferences we replicate:
//
//   preLinebreakOnExport: true     → marked's breaks:true (line ~216). Single
//                                    newlines in markdown become <br> in output.
//                                    This is Typora's default for new installs.
//
//   strict_mode: true              → Our marked extensions (==highlight==,
//                                    ~subscript~, ^superscript^) still work
//                                    because they are parser extensions, not
//                                    loose-mode behaviors. Typora's strict mode
//                                    disables some edge-case markdown behaviors
//                                    (e.g., no lazy continuation); marked's GFM
//                                    mode is similarly strict by default.
//
//   theme: "Github"                → We ship both pixyll and github themes.
//   darkTheme: "Pixyll"            → Default --theme is pixyll. User can pass
//                                    --theme github to match Typora's light theme.
//
//   WebAutomaticDashSubstitution   → false. We do NOT apply smart dash
//   WebAutomaticQuoteSubstitution  → false. We do NOT apply smart quote
//                                    substitution. Both systems output literal
//                                    characters from the markdown source.
//
//   enable_inline_math: false      → We ENABLE math anyway (see INTENTIONAL
//                                    DIFFERENCES). This pref is false on this
//                                    machine but our pipeline supports it because
//                                    the user may toggle it or use math in docs.
//
// INTENTIONAL DIFFERENCES — Features where we do MORE than Typora's config:
//
//   Math (KaTeX):    We always enable $inline$ and $$block$$ math rendering.
//                    Typora has enable_inline_math=false on this machine, so
//                    Typora would show raw LaTeX. We render it. This is
//                    intentional — our pipeline should handle any valid markdown.
//
//   GFM Alerts:      We render > [!NOTE], > [!TIP], > [!WARNING], etc. as
//                    styled callout boxes. Typora does not support this syntax.
//
//   Footnotes:       We render [^1] footnotes via marked-footnote. Typora
//                    renders these too (built-in), but our output structure
//                    differs slightly (marked-footnote uses <section> with
//                    backlinks; Typora uses its own DOM).
//
//   Mermaid:         We render mermaid code blocks as SVG diagrams via the
//                    mermaid.js library injected into the Chromium page.
//                    Typora also renders Mermaid natively. Output is equivalent.
//
//   Emoji:           We convert :shortcode: to Unicode glyphs via node-emoji.
//                    Typora does this too. Equivalent behavior.
//
//   TOC:             We replace [toc] with a generated table of contents.
//                    Typora does this too. Equivalent behavior.
//
// NON-RENDERING preferences (no pipeline impact):
//
//   copy_markdown_by_default: true   — clipboard behavior, editor-only
//   use_seamless_window: true        — window chrome, editor-only
//   useSeparateDarkTheme: true       — theme switching, editor-only
//   schemeAwareness: false           — system appearance, editor-only
//   send_usage_info: true            — telemetry, editor-only
//
// See README.md in this directory for the full feature comparison and
// rendering audit results.
//

import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync, unlinkSync, watch as fsWatch } from 'node:fs';
import { resolve, dirname, basename, extname, join, relative } from 'node:path';
import { homedir, tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'node:http';

// ── Argument parsing ──────────────────────────────────────────────────────────

const { values: opts, positionals } = parseArgs({
  allowPositionals: true,
  allowNegative: true,
  options: {
    stylesheet: { type: 'string', short: 's' },
    theme:      { type: 'string', short: 't', default: 'pixyll' },
    width:      { type: 'string', short: 'w', default: '1400' },
    engine:     { type: 'string', short: 'e', default: 'chromium' },
    'front-matter': { type: 'string', default: 'strip' },
    html:       { type: 'boolean', default: false },
    pageless:   { type: 'boolean', default: true },
    'page-numbers': { type: 'boolean', default: true },
    header:     { type: 'string' },
    footer:     { type: 'string' },
    serve:      { type: 'boolean', default: false },
    host:       { type: 'string', default: 'localhost' },
    output:     { type: 'string', short: 'o' },
    mobile:     { type: 'boolean', short: 'm', default: false },
    'rich-highlighting': { type: 'boolean', default: false },
    help:       { type: 'boolean', short: 'h', default: false },
  },
});

if (opts.help || positionals.length === 0) {
  console.log(`Usage: node markdown-kit.js <input.md> [options]

Options:
  --theme, -t <name>        Theme: pixyll (default), github, gothic, newsprint, night, whitey
  --stylesheet, -s <path>   Custom CSS (overrides --theme)
  --width, -w <pixels>      Page width in pixels (default: 1400)
  --engine, -e <name>       PDF engine: chromium (default) or webkit
  --front-matter <mode>     How to handle YAML front matter: strip (default), render, raw
  --no-pageless             Use normal A4 pages instead of single continuous page
  --no-page-numbers         Disable page numbers in paged mode (enabled by default)
  --header <template>       Custom header HTML for paged mode
  --footer <template>       Custom footer HTML for paged mode (overrides default page numbers)
  --serve                   Start a live preview server instead of generating PDF
  --host <addr>             Bind address: localhost (default) or 0.0.0.0 for network
  -o, --output <path>       Output file path (default: input.pdf alongside input)
  --html                    Output self-contained HTML instead of PDF
  -m, --mobile              Optimize for phone reading (430px, smaller fonts)
  --rich-highlighting       Color built-ins and types distinctly (print, Optional, List, etc.)
  --help, -h                Show this help

Front matter modes:
  strip     Remove front matter from output (default). Uses title field for PDF metadata.
  render    Show front matter as a styled metadata header at the top of the document.
  raw       Show front matter as a YAML code block at the top of the document.

Header/footer templates:
  Use {{pageNumber}}, {{totalPages}} as placeholders in --header/--footer strings.
  YAML front matter "header" and "footer" fields also work (CLI flags take priority).
  Example: --footer "Page {{pageNumber}} of {{totalPages}}"
  Default footer in paged mode: centered "Page X of Y" in theme-matching font.

Engines:
  chromium    Headless Chromium via Puppeteer (default, cross-platform)
  webkit      macOS native WKWebView — same engine as Typora (macOS only)

Output: <input>.pdf alongside the input file (or live preview with --serve).`);
  process.exit(opts.help ? 0 : 1);
}

const inputPath = resolve(positionals[0]);
if (!existsSync(inputPath)) {
  console.error(`Error: file not found: ${inputPath}`);
  process.exit(1);
}
let pageWidth = parseInt(opts.width, 10);
if (isNaN(pageWidth) || pageWidth <= 0) {
  console.error(`Error: --width must be a positive integer, got '${opts.width}'`);
  process.exit(1);
}
// --mobile preset: override width to 430px (iPhone) unless --width was explicitly set
if (opts.mobile && opts.width === '1400') {
  pageWidth = 430;
}
const engine = opts.engine;
if (engine !== 'chromium' && engine !== 'webkit') {
  console.error(`Error: --engine must be 'chromium' or 'webkit', got '${engine}'`);
  process.exit(1);
}

const frontMatterMode = opts['front-matter'];
if (!['strip', 'render', 'raw'].includes(frontMatterMode)) {
  console.error(`Error: --front-matter must be 'strip', 'render', or 'raw', got '${frontMatterMode}'`);
  process.exit(1);
}

// Resolve stylesheet: --stylesheet overrides --theme
const scriptDir = dirname(fileURLToPath(import.meta.url));
const stylesheetPath = opts.stylesheet
  ? resolve(opts.stylesheet)
  : join(scriptDir, 'themes', opts.theme, 'theme.css');

// ── Dependency bootstrap ──────────────────────────────────────────────────────
// Install puppeteer + marked into a persistent cache dir on first run.
// Uses createRequire() to load them from the cache, bypassing the ESM
// resolution bug with npx on Node 24.

const CACHE_DIR = join(homedir(), '.claude-workspace', 'tools', 'markdown-kit');
const DEPS = {
  puppeteer: '^24',
  marked: '^15',
  'marked-highlight': '^2',
  'highlight.js': '^11',
  mermaid: '^11',
  katex: '^0.16',
  'marked-katex-extension': '^5',
  'marked-gfm-heading-id': '^4',
  'marked-emoji': '^2',
  'node-emoji': '^2',
  'marked-footnote': '^1',
  'marked-alert': '^2',
  'gray-matter': '^4',
  'pdf-lib': '^1',
  '@hpcc-js/wasm-graphviz': '^1',
  'vega': '^5',
  'vega-lite': '^5',
};

function ensureDeps() {
  const nmDir = join(CACHE_DIR, 'node_modules');
  const allInstalled = Object.keys(DEPS).every(pkg =>
    existsSync(join(nmDir, pkg, 'package.json'))
  );

  if (!allInstalled) {
    console.log('Installing dependencies (one-time)...');
    mkdirSync(CACHE_DIR, { recursive: true });
    const pkgs = Object.entries(DEPS).map(([n, v]) => `${n}@${v}`).join(' ');
    execSync(`npm install --prefix "${CACHE_DIR}" ${pkgs}`, {
      stdio: ['pipe', 'pipe', 'inherit'],
    });
    console.log('Dependencies cached at', CACHE_DIR);
  }

  return createRequire(join(nmDir, '_anchor.js'));
}

const require = ensureDeps();
const puppeteer = require('puppeteer');
const { marked } = require('marked');
const { markedHighlight } = require('marked-highlight');
const hljs = require('highlight.js');
const markedKatex = require('marked-katex-extension');
const { gfmHeadingId, getHeadingList } = require('marked-gfm-heading-id');
const { markedEmoji } = require('marked-emoji');
const nodeEmoji = require('node-emoji');
const markedFootnote = require('marked-footnote');
const markedAlert = require('marked-alert');
const matter = require('gray-matter');
const { PDFDocument } = require('pdf-lib');

// Load mhchem extension for KaTeX — enables \ce{} chemical formula notation.
// Must be loaded after katex is available; it registers itself automatically.
require('katex/contrib/mhchem');

// Graphviz/DOT diagram support — WASM-based renderer for server-side SVG
const { Graphviz } = require('@hpcc-js/wasm-graphviz');

// Vega-Lite chart support — server-side SVG rendering
const vl = require('vega-lite');
const vega = require('vega');

// Resolve the mermaid browser bundle path for injection into Puppeteer pages
const mermaidPkgDir = dirname(require.resolve('mermaid/package.json'));
const mermaidBundlePath = join(mermaidPkgDir, 'dist', 'mermaid.min.js');

// ── Markdown to HTML ──────────────────────────────────────────────────────────

const inputDir = dirname(inputPath);
const outputExt = opts.html ? '.html' : '.pdf';
const outputPath = opts.output
  ? resolve(opts.output)
  : join(inputDir, basename(inputPath, extname(inputPath)) + outputExt);
// Create parent directories for custom output paths
if (opts.output) {
  mkdirSync(dirname(outputPath), { recursive: true });
}

// Configure marked for GFM + syntax highlighting
// breaks: true matches Typora's preLinebreakOnExport=1 (default for new installs)
marked.setOptions({ gfm: true, breaks: true });
marked.use(markedHighlight({
  langPrefix: 'hljs language-',
  highlight(code, lang) {
    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
    return hljs.highlight(code, { language }).value;
  }
}));

// Enable math rendering: $inline$ and $$block$$ via KaTeX
// marked-katex-extension default export may be the function itself or { default: fn }
const markedKatexFn = typeof markedKatex === 'function' ? markedKatex : markedKatex.default;
marked.use(markedKatexFn({ throwOnError: false }));

// Footnotes: [^1] references and [^1]: definitions at bottom of document
// marked-footnote default export may be the function itself or { default: fn }
const markedFootnoteFn = typeof markedFootnote === 'function' ? markedFootnote : markedFootnote.default;
marked.use(markedFootnoteFn());

// GFM Alerts: > [!NOTE], > [!TIP], > [!WARNING], > [!CAUTION], > [!IMPORTANT]
// Transforms blockquotes with alert syntax into styled <div> callout boxes
const markedAlertFn = typeof markedAlert === 'function' ? markedAlert : markedAlert.default;
marked.use(markedAlertFn());

// ── Typora extensions ─────────────────────────────────────────────────────────

// GFM heading IDs (required for TOC anchor links)
marked.use(gfmHeadingId());

// ==highlight== -> <mark>highlight</mark>
// Custom inline extension for Typora's highlight syntax
const highlightExtension = {
  name: 'highlight',
  level: 'inline',
  start(src) { return src.indexOf('=='); },
  tokenizer(src) {
    const match = src.match(/^==([^=]+)==/);
    if (match) {
      return { type: 'highlight', raw: match[0], text: this.lexer.inlineTokens(match[1]) };
    }
  },
  renderer(token) {
    return `<mark>${this.parser.parseInline(token.text)}</mark>`;
  }
};

// ~subscript~ -> <sub>subscript</sub>
// Single tilde only — GFM ~~strikethrough~~ (double tilde) is handled by marked core.
// Match ~text~ but NOT ~~text~~ (negative lookahead/lookbehind).
const subscriptExtension = {
  name: 'subscript',
  level: 'inline',
  start(src) {
    const m = src.match(/(?<![~\\])~/);
    if (m && src[m.index + 1] !== '~') return m.index;
    return -1;
  },
  tokenizer(src) {
    // Match single ~ pairs: ~text~ but not ~~text~~
    const match = src.match(/^~(?!~)([^~\n]+)~(?!~)/);
    if (match) {
      return { type: 'subscript', raw: match[0], text: match[1] };
    }
  },
  renderer(token) {
    return `<sub>${token.text}</sub>`;
  }
};

// ^superscript^ -> <sup>superscript</sup>
const superscriptExtension = {
  name: 'superscript',
  level: 'inline',
  start(src) { return src.indexOf('^'); },
  tokenizer(src) {
    const match = src.match(/^\^([^^]+)\^/);
    if (match) {
      return { type: 'superscript', raw: match[0], text: match[1] };
    }
  },
  renderer(token) {
    return `<sup>${token.text}</sup>`;
  }
};

// :::type ... ::: -> <div class="custom-container type"> ... </div>
// Custom container blocks for callout/admonition content beyond GFM alerts.
// Supports any type name: :::info, :::warning, :::success, :::danger, etc.
const containerExtension = {
  name: 'container',
  level: 'block',
  start(src) { return src.match(/^:::[a-z][\w-]*/m)?.index; },
  tokenizer(src) {
    const match = src.match(/^:::([a-z][\w-]*)\n([\s\S]*?)\n:::\s*(?:\n|$)/);
    if (match) {
      const token = {
        type: 'container',
        raw: match[0],
        containerType: match[1],
        tokens: [],
      };
      // Parse the body content into child tokens inline
      this.lexer.blockTokens(match[2], token.tokens);
      return token;
    }
  },
  renderer(token) {
    const inner = this.parser.parse(token.tokens);
    const typeLabel = token.containerType.charAt(0).toUpperCase() + token.containerType.slice(1);
    return `<div class="custom-container ${token.containerType}">\n<p class="custom-container-title">${typeLabel}</p>\n${inner}\n</div>\n`;
  },
};

marked.use({ extensions: [highlightExtension, subscriptExtension, superscriptExtension, containerExtension] });

// :emoji_name: -> unicode emoji character
// Build emoji map from node-emoji's search (covers all GitHub/Typora shortcodes)
const emojiMap = {};
for (const { name, emoji } of nodeEmoji.search('')) {
  emojiMap[name] = emoji;
}
// Add common GitHub/Typora aliases missing from node-emoji
const emojiAliases = {
  thumbsup: '+1', thumbs_up: '+1', thumbsdown: '-1', thumbs_down: '-1',
  shipit: 'ship', satisfied: 'laughing', hankey: 'poop', poop: 'hankey',
};
for (const [alias, canonical] of Object.entries(emojiAliases)) {
  if (!emojiMap[alias] && emojiMap[canonical]) {
    emojiMap[alias] = emojiMap[canonical];
  } else if (!emojiMap[alias]) {
    const found = nodeEmoji.get(canonical);
    if (found) emojiMap[alias] = found;
  }
}
marked.use(markedEmoji({ emojis: emojiMap, renderer: (token) => token.emoji }));

// ── Reusable pipeline functions ────────────────────────────────────────────────

function generateTocHtml(headings) {
  if (!headings || headings.length === 0) return '';
  const items = headings.map(({ id, raw, level }) => {
    const indent = (level - 1) * 20;
    return `<li style="margin-left:${indent}px"><a href="#${id}">${raw}</a></li>`;
  });
  return `<div class="md-toc"><ul>\n${items.join('\n')}\n</ul></div>`;
}

function generateFrontMatterHtml(mode, data, yaml, hasFM) {
  if (!hasFM || mode === 'strip') return '';

  if (mode === 'raw') {
    const escapedYaml = yaml
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    return `<div class="front-matter-raw">
<div class="front-matter-label">Front Matter</div>
<pre><code class="language-yaml">${escapedYaml}</code></pre>
</div>\n`;
  }

  if (mode === 'render') {
    const displayFields = [
      { key: 'title', label: 'Title' },
      { key: 'author', label: 'Author' },
      { key: 'date', label: 'Date' },
      { key: 'tags', label: 'Tags' },
      { key: 'description', label: 'Description' },
      { key: 'subtitle', label: 'Subtitle' },
      { key: 'version', label: 'Version' },
      { key: 'status', label: 'Status' },
      { key: 'category', label: 'Category' },
      { key: 'categories', label: 'Categories' },
      { key: 'lang', label: 'Language' },
      { key: 'license', label: 'License' },
    ];

    const items = [];
    for (const { key, label } of displayFields) {
      if (data[key] == null) continue;
      let value = data[key];
      if (Array.isArray(value)) value = value.join(', ');
      const escaped = String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      items.push(`  <div class="front-matter-field"><dt>${label}</dt><dd>${escaped}</dd></div>`);
    }

    if (items.length === 0) return '';

    return `<div class="front-matter-header">
<dl>
${items.join('\n')}
</dl>
</div>\n`;
  }

  return '';
}

// Resolve image paths: convert relative src to file:// URLs (for PDF mode)
function resolveImagePaths(html, baseDir) {
  return html.replace(
    /(<img\s[^>]*src=")([^"]+)(")/gi,
    (match, pre, src, post) => {
      if (/^(https?:|data:|file:)/i.test(src)) return match;
      const absPath = resolve(baseDir, src);
      const encodedPath = absPath.split('/').map(s => encodeURIComponent(s)).join('/');
      return `${pre}file://${encodedPath}${post}`;
    }
  );
}

// Resolve image paths: convert relative src to /static/ URLs (for serve mode)
function resolveImagePathsForServe(html) {
  return html.replace(
    /(<img\s[^>]*src=")([^"]+)(")/gi,
    (match, pre, src, post) => {
      if (/^(https?:|data:|file:)/i.test(src)) return match;
      // Keep relative paths as-is but route through /static/
      return `${pre}/static/${encodeURI(src)}${post}`;
    }
  );
}

// Load highlight.js CSS theme — custom theme with richer semantic coloring
// The stock github.css collapses many token types into the same color (only 7 colors
// for 20+ token classes). Our hljs-typora.css assigns 12 distinct colors, matching
// the rich coloring that Typora's CodeMirror engine produces.
const hljsCssPath = join(scriptDir, 'hljs-typora.css');
let hljsCss = existsSync(hljsCssPath) ? readFileSync(hljsCssPath, 'utf-8') : '';
// When --rich-highlighting is off (default), built-in functions (print, len, sum)
// and type identifiers (Optional, List, Dict) render as regular text — matching
// PyCharm/VS Code default behavior. Opt in with --rich-highlighting for distinct
// semantic coloring (gold built-ins, dark orange types).
if (!opts['rich-highlighting']) {
  hljsCss += '\n.hljs-built_in, .hljs-type { color: inherit !important; }\n';
}

// Load KaTeX CSS from the cached package
const katexCssPath = join(CACHE_DIR, 'node_modules', 'katex', 'dist', 'katex.css');
const katexFontsDir = join(CACHE_DIR, 'node_modules', 'katex', 'dist', 'fonts');
function loadKatexCss(forServe) {
  if (!existsSync(katexCssPath)) return '';
  let css = readFileSync(katexCssPath, 'utf-8');
  if (forServe) {
    // Route font URLs through our HTTP server
    css = css.replace(/url\(fonts\//g, 'url(/katex-fonts/');
  } else {
    // Use file:// URLs for Puppeteer/PDF mode
    css = css.replace(
      /url\(fonts\//g,
      () => {
        const encodedDir = katexFontsDir.split('/').map(s => encodeURIComponent(s)).join('/');
        return `url(file://${encodedDir}/`;
      }
    );
  }
  return css;
}

// Load theme stylesheet
function loadThemeCss(forServe) {
  if (!existsSync(stylesheetPath)) {
    console.warn('Warning: stylesheet not found:', stylesheetPath);
    return '';
  }
  let css = readFileSync(stylesheetPath, 'utf-8');
  const cssDir = dirname(stylesheetPath);
  if (forServe) {
    // Route relative url() references through our HTTP server
    css = css.replace(
      /url\(['"]?(\.\/?[^'")]+)['"]?\)/g,
      (match, relPath) => {
        return `url('/theme-assets/${encodeURI(relPath)}')`;
      }
    );
  } else {
    // Use file:// URLs for Puppeteer/PDF mode
    css = css.replace(
      /url\(['"]?(\.\/?[^'")]+)['"]?\)/g,
      (match, relPath) => {
        const absPath = resolve(cssDir, relPath);
        const encodedPath = absPath.split('/').map(s => encodeURIComponent(s)).join('/');
        return `url('file://${encodedPath}')`;
      }
    );
  }
  return css;
}

// ── Header/footer templates for paged mode ──────────────────────────────────
//
// Puppeteer's page.pdf() supports displayHeaderFooter with headerTemplate and
// footerTemplate. These render inside the margin area of each page using a
// separate Chromium renderer with NO access to external stylesheets — all CSS
// must be inline. Available CSS classes for dynamic values:
//   .pageNumber, .totalPages, .date, .title, .url
//
// Theme font mapping — each theme's primary font for visual consistency:
const THEME_FONTS = {
  pixyll:    '"Merriweather", "PT Serif", Georgia, serif',
  github:    '"Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif',
  gothic:    '"Didact Gothic", "Century Gothic", sans-serif',
  newsprint: '"PT Serif", "Times New Roman", Times, serif',
  night:     '"Helvetica Neue", Helvetica, Arial, sans-serif',
  whitey:    '"Vollkorn", Palatino, Times, serif',
};

/**
 * Convert user-friendly {{pageNumber}} / {{totalPages}} placeholders
 * to Puppeteer's <span class="pageNumber"></span> format.
 */
function expandTemplatePlaceholders(template) {
  return template
    .replace(/\{\{pageNumber\}\}/gi, '<span class="pageNumber"></span>')
    .replace(/\{\{totalPages\}\}/gi, '<span class="totalPages"></span>')
    .replace(/\{\{date\}\}/gi, '<span class="date"></span>')
    .replace(/\{\{title\}\}/gi, '<span class="title"></span>');
}

/**
 * Build a Puppeteer header or footer template string with inline CSS.
 * @param {string} content - The text/HTML content to display
 * @param {string} fontFamily - CSS font-family string
 * @param {object} options
 * @param {string} options.fontSize - Font size (default: '9px')
 * @param {string} options.color - Text color (default: '#888')
 * @param {string} options.align - Text alignment (default: 'center')
 * @param {string} options.padding - Padding (default: '0 20mm')
 * @returns {string} Complete HTML template for Puppeteer
 */
function buildTemplate(content, fontFamily, { fontSize = '9px', color = '#888', align = 'center', padding = '0 20mm' } = {}) {
  const expanded = expandTemplatePlaceholders(content);
  return `<div style="font-family: ${fontFamily}; font-size: ${fontSize}; color: ${color}; text-align: ${align}; width: 100%; padding: ${padding}; box-sizing: border-box;">${expanded}</div>`;
}

/**
 * Resolve the header and footer templates for paged PDF mode.
 *
 * Priority order (highest to lowest):
 *   1. CLI flags: --header, --footer
 *   2. YAML front matter: header, footer fields
 *   3. Defaults: no header; "Page X of Y" footer (if page numbers enabled)
 *
 * @param {object} frontMatter - Parsed YAML front matter
 * @param {string} themeName - Current theme name (for font lookup)
 * @returns {{ headerTemplate: string, footerTemplate: string, displayHeaderFooter: boolean }}
 */
function resolveHeaderFooterTemplates(frontMatter, themeName) {
  const fontFamily = THEME_FONTS[themeName] || THEME_FONTS.pixyll;
  const pageNumbersEnabled = opts['page-numbers'];

  // Resolve header content: CLI > front matter > none
  let headerContent = opts.header || null;
  if (!headerContent && frontMatter && frontMatter.header) {
    headerContent = String(frontMatter.header);
  }

  // Resolve footer content: CLI > front matter > default page numbers
  let footerContent = opts.footer || null;
  if (!footerContent && frontMatter && frontMatter.footer) {
    footerContent = String(frontMatter.footer);
  }
  if (!footerContent && pageNumbersEnabled) {
    footerContent = 'Page {{pageNumber}} of {{totalPages}}';
  }

  const hasHeader = !!headerContent;
  const hasFooter = !!footerContent;

  if (!hasHeader && !hasFooter) {
    return { headerTemplate: '', footerTemplate: '', displayHeaderFooter: false };
  }

  // Build templates with inline CSS
  // Puppeteer requires a non-empty template even if unused (empty string hides it)
  const headerTemplate = hasHeader
    ? buildTemplate(headerContent, fontFamily, { padding: '0 20mm', fontSize: '9px' })
    : '<span></span>';  // invisible placeholder

  const footerTemplate = hasFooter
    ? buildTemplate(footerContent, fontFamily, { padding: '0 20mm' })
    : '<span></span>';

  return { headerTemplate, footerTemplate, displayHeaderFooter: true };
}

/**
 * Build complete HTML from the input markdown file.
 * @param {object} options
 * @param {boolean} options.forServe - If true, use HTTP-served asset paths instead of file:// URLs
 * @returns {Promise<string>} Complete HTML document
 */
async function buildHtml({ forServe = false } = {}) {
  const mdSource = readFileSync(inputPath, 'utf-8');

  // Parse front matter
  const parsed = matter(mdSource);
  let mdContent = parsed.content;
  const fm = parsed.data;
  const rawYaml = parsed.matter;
  const hasFM = Object.keys(fm).length > 0;

  // ── Variable substitution (Feature: {{key}} placeholders) ──────────────────
  // Replace {{key}} with values from front matter before markdown parsing.
  // Only touches keys that exist in front matter; unmatched {{key}} are left as-is.
  if (hasFM) {
    mdContent = mdContent.replace(/\{\{(\w+)\}\}/g, (match, key) => {
      return fm[key] != null ? String(fm[key]) : match;
    });
  }

  // Parse markdown to HTML
  const htmlBody = marked.parse(mdContent);

  // TOC generation: replace [toc] placeholders
  let processedBody = htmlBody;
  const tocHtml = generateTocHtml(getHeadingList());
  processedBody = processedBody.replace(/<p>\s*\[toc\]\s*<\/p>/gi, tocHtml);

  // ── Server-side diagram/chart rendering ────────────────────────────────────
  // Graphviz DOT and Vega-Lite code blocks are rendered to SVG server-side
  // before the HTML is passed to Puppeteer. This avoids browser-side WASM
  // loading and keeps these renders fast and deterministic.

  // Graphviz/DOT: <pre><code class="hljs language-dot"> or language-graphviz
  const dotPattern = /<pre><code class="(?:hljs )?language-(?:dot|graphviz)">([\s\S]*?)<\/code><\/pre>/g;
  const dotBlocks = [...processedBody.matchAll(dotPattern)];
  if (dotBlocks.length > 0) {
    const graphviz = await Graphviz.load();
    for (const block of dotBlocks) {
      const dotSource = block[1]
        .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
        .replace(/&quot;/g, '"').replace(/&#39;/g, "'");
      try {
        const svg = graphviz.dot(dotSource);
        processedBody = processedBody.replace(block[0],
          `<div class="graphviz-diagram">${svg}</div>`);
      } catch (e) {
        console.warn('Warning: Graphviz render failed:', e.message);
      }
    }
  }

  // Vega-Lite: <pre><code class="hljs language-vega-lite"> or language-chart
  const vlPattern = /<pre><code class="(?:hljs )?language-(?:vega-lite|chart)">([\s\S]*?)<\/code><\/pre>/g;
  const vlBlocks = [...processedBody.matchAll(vlPattern)];
  for (const block of vlBlocks) {
    const specText = block[1]
      .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&')
      .replace(/&quot;/g, '"').replace(/&#39;/g, "'");
    try {
      const spec = JSON.parse(specText);
      const compiled = vl.compile(spec);
      const view = new vega.View(vega.parse(compiled.spec), { renderer: 'none' });
      const svg = await view.toSVG();
      view.finalize();
      processedBody = processedBody.replace(block[0],
        `<div class="vega-lite-chart">${svg}</div>`);
    } catch (e) {
      console.warn('Warning: Vega-Lite render failed:', e.message);
    }
  }

  // Resolve image paths
  let resolvedBody = forServe
    ? resolveImagePathsForServe(processedBody)
    : resolveImagePaths(processedBody, inputDir);

  // Strip inline page-break styles in pageless mode (only inside style= attributes,
  // NOT inside <code> tags where they might be documentation text)
  if (opts.pageless) {
    resolvedBody = resolvedBody.replace(
      /(<[^>]+style="[^"]*?)page-break-(before|after|inside)\s*:\s*[^;"]+;?/gi,
      '$1'
    );
    resolvedBody = resolvedBody.replace(
      /(<[^>]+style="[^"]*?)break-(before|after|inside)\s*:\s*[^;"]+;?/gi,
      '$1'
    );
  }

  // Extract title
  const titleMatch = mdContent.match(/^#\s+(.+)$/m);
  const rawTitle = (hasFM && fm.title)
    ? String(fm.title)
    : (titleMatch ? titleMatch[1].trim() : basename(inputPath, extname(inputPath)));
  const docTitle = rawTitle.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Front matter HTML
  const fmHtml = generateFrontMatterHtml(frontMatterMode, fm, rawYaml, hasFM);
  if (fmHtml) resolvedBody = fmHtml + resolvedBody;

  // Load CSS with appropriate URL resolution
  const themeCss = loadThemeCss(forServe);
  const kCss = loadKatexCss(forServe);

  // Serve mode extras: client-side mermaid + SSE live-reload
  const serveScripts = forServe ? `
  <script src="/mermaid.min.js"></script>
  <script>
    mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });
    // Convert highlighted mermaid code blocks to renderable divs
    document.querySelectorAll('pre > code.language-mermaid').forEach(code => {
      const pre = code.parentElement;
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      pre.replaceWith(div);
    });
    mermaid.run({ querySelector: '.mermaid' });
  </script>
  <script>
    const evtSource = new EventSource('/events');
    evtSource.addEventListener('reload', () => location.reload());
  </script>` : '';

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  ${forServe ? '<meta name="viewport" content="width=device-width, initial-scale=1">' : ''}
  <title>${docTitle}</title>
  <style>${themeCss}</style>
  <style>${hljsCss}</style>
  <style>${kCss}</style>
  <style>
    /* Pageless overrides — no page breaks, continuous flow */
    * {
      page-break-inside: auto !important;
      page-break-after: auto !important;
      page-break-before: auto !important;
      break-after: auto !important;
      break-before: auto !important;
      break-inside: auto !important;
    }
    body {
      ${forServe ? `width: 100%; max-width: ${pageWidth}px;` : `width: ${pageWidth}px; max-width: ${pageWidth}px;`}
      margin: 0 auto;
      padding: ${forServe ? '20px' : '40px 40px 0px 40px'};
      padding-bottom: 0px !important;
      margin-bottom: 0px !important;
      box-sizing: border-box;
    }
    img {
      max-width: 100%;
      height: auto;
    }
    ${forServe ? `
    /* Mobile responsive overrides for serve mode */
    @media (max-width: 767px) {
      html { font-size: 14px; }
      body { padding: 16px; font-size: 1.1rem; }
      h1 { font-size: 1.8rem; }
      h2 { font-size: 1.4rem; }
      h3 { font-size: 1.2rem; }
      h4, h5, h6 { font-size: 1rem; }
      p, li { font-size: 1rem; line-height: 1.6; }
      pre, .katex-display, table { overflow-x: auto; -webkit-overflow-scrolling: touch; }
      table { display: block; font-size: 0.9rem; }
      .graphviz-diagram svg, .mermaid svg { max-width: 100%; height: auto; }
    }` : ''}
    ${(!forServe && opts.mobile) ? `
    /* Mobile PDF overrides — smaller fonts and spacing for phone-sized pages */
    html { font-size: 14px; }
    body { font-size: 1.1rem; padding: 16px !important; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.2rem !important; }
    h4, h5, h6 { font-size: 1rem !important; }
    p, li { font-size: 1rem !important; line-height: 1.6 !important; }
    table { font-size: 0.8rem !important; word-break: break-word; }
    pre { font-size: 0.8rem !important; }
    .graphviz-diagram svg, .mermaid svg { max-width: 100%; height: auto; }
    ` : ''}
    /* TOC styling */
    .md-toc {
      border: 1px solid #ddd;
      border-radius: 4px;
      padding: 12px 20px;
      margin: 16px 0;
      background: #f9f9f9;
    }
    .md-toc ul {
      list-style: none;
      padding: 0;
      margin: 0;
    }
    .md-toc li {
      line-height: 1.8;
    }
    .md-toc a {
      text-decoration: none;
      color: #4183c4;
    }
    .md-toc a:hover {
      text-decoration: underline;
    }
    /* Highlight (==text==) */
    mark {
      background-color: #fff3a8;
      padding: 0.1em 0.2em;
      border-radius: 2px;
    }
    /* Custom containers (:::type ... :::) */
    .custom-container {
      border-left: 4px solid #ccc;
      border-radius: 4px;
      padding: 12px 16px;
      margin: 16px 0;
      background: #f9f9f9;
    }
    .custom-container-title {
      font-weight: 700;
      margin: 0 0 8px 0;
    }
    .custom-container.info { border-left-color: #2196F3; background: #e3f2fd; }
    .custom-container.info .custom-container-title { color: #1565C0; }
    .custom-container.success { border-left-color: #4CAF50; background: #e8f5e9; }
    .custom-container.success .custom-container-title { color: #2E7D32; }
    .custom-container.warning { border-left-color: #FF9800; background: #fff3e0; }
    .custom-container.warning .custom-container-title { color: #E65100; }
    .custom-container.danger { border-left-color: #F44336; background: #ffebee; }
    .custom-container.danger .custom-container-title { color: #C62828; }
    .custom-container.tip { border-left-color: #009688; background: #e0f2f1; }
    .custom-container.tip .custom-container-title { color: #00695C; }
    /* Graphviz diagram containers */
    .graphviz-diagram {
      margin: 16px 0;
      text-align: center;
    }
    .graphviz-diagram svg {
      max-width: 100%;
      height: auto;
    }
    /* Vega-Lite chart containers */
    .vega-lite-chart {
      margin: 16px 0;
      text-align: center;
    }
    .vega-lite-chart svg {
      max-width: 100%;
      height: auto;
    }
  </style>
</head>
<body>
${resolvedBody}
</body>
${serveScripts}
</html>`;
}

// Build the initial HTML (used by both PDF and serve modes)
const fullHtml = await buildHtml({ forServe: false });

// Capture heading list after marked.parse() runs inside buildHtml().
// getHeadingList() returns the headings from the most recent parse, so
// this must come before any subsequent marked.parse() call.
const parsedHeadings = getHeadingList();

// Parse front matter once at module level for PDF metadata injection.
// buildHtml() also parses it internally, but we need the raw object here
// for post-processing after PDF generation.
const parsedFrontMatter = matter(readFileSync(inputPath, 'utf-8')).data;

// ── Internal link collection and PDF post-processing ──────────────────────────
//
// Chromium's page.pdf() generates internal link annotations using named
// destinations (/Dest /name), but only for a subset of anchor links. Many
// internal links (especially those from TOC, footnotes, and cross-references)
// are missing. We work around this by:
// 1. Collecting all internal link sources and anchor targets via page.evaluate()
// 2. Generating the PDF normally
// 3. Post-processing with Apple PDFKit (via compiled Swift tool) to add link
//    annotations with inline /Dest arrays — the format macOS Preview requires
//
// Why Apple PDFKit instead of pdf-lib (JavaScript)?
//   pdf-lib produces annotations with only /A (GoTo action) entries, which some
//   PDF viewers (notably macOS Preview.app) do not navigate correctly. Apple
//   PDFKit's annotation.destination property writes an inline /Dest array
//   directly on the annotation dictionary, matching the format that Chromium's
//   own native internal links use and that Preview handles reliably.
//
// The Swift tool (add-links) also handles:
//   - Removing Chromium's existing internal link annotations (--remove-existing)
//   - Coordinate mapping (CSS pixels → PDF points with Y-axis flip)
//   - Both pageless (single-page) and paged (multi-page A4) modes

/**
 * JavaScript to inject into the page to collect internal link/anchor data.
 * Must be called after print media emulation and layout reflow.
 * Returns { links: [...], anchors: {id: {x,y,w,h}}, docHeight: number }
 */
const COLLECT_LINKS_JS = `(() => {
  const links = [];
  const anchors = {};

  // Collect all elements with IDs (potential link targets)
  document.querySelectorAll('[id]').forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.width > 0 || rect.height > 0) {
      anchors[el.id] = {
        x: rect.left + window.scrollX,
        y: rect.top + window.scrollY,
        w: rect.width,
        h: rect.height
      };
    }
  });

  // Collect all internal links (href starting with #)
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    const rect = a.getBoundingClientRect();
    const rawHref = a.getAttribute('href').substring(1);
    const targetId = decodeURIComponent(rawHref);
    if (rect.width > 0 && rect.height > 0) {
      links.push({
        sx: rect.left + window.scrollX,
        sy: rect.top + window.scrollY,
        sw: rect.width,
        sh: rect.height,
        targetId: targetId
      });
    }
  });

  return JSON.stringify({
    links,
    anchors,
    docHeight: document.body.scrollHeight,
    viewportWidth: window.innerWidth
  });
})()`;

/**
 * Post-process a PDF to add clickable internal link annotations using
 * Apple PDFKit (via the compiled add-links Swift tool).
 *
 * Shells out to the add-links binary which uses PDFKit.framework — the same
 * framework macOS Preview.app and iOS use to read PDF annotations. This
 * guarantees cross-viewer compatibility that pdf-lib (JavaScript) cannot
 * achieve due to subtle serialization incompatibilities.
 *
 * @param {string} pdfPath - Path to the PDF file to modify in-place
 * @param {object} linkData - { links, anchors, docHeight } from COLLECT_LINKS_JS
 * @param {object} opts - { pageless: boolean }
 * @returns {{ added: number, removed: number, skipped: number }}
 */
function addInternalLinkAnnotations(pdfPath, linkData, { pageless }) {
  const { links } = linkData;
  if (!links || links.length === 0) return { added: 0, removed: 0, skipped: 0 };

  // Locate the add-links binary (compiled Swift tool using Apple PDFKit)
  const addLinksBin = join(scriptDir, 'add-links');
  if (!existsSync(addLinksBin)) {
    console.warn('Warning: add-links binary not found at', addLinksBin);
    console.warn('Compile it with: xcrun swiftc -o add-links add-links.swift -framework PDFKit -framework AppKit');
    return { added: 0, removed: 0, skipped: 0 };
  }

  // Write link data to a temp JSON file
  const tmpJsonPath = join(tmpdir(), `pageless-links-${process.pid}.json`);
  writeFileSync(tmpJsonPath, JSON.stringify(linkData));

  try {
    const args = [
      JSON.stringify(pdfPath),
      JSON.stringify(tmpJsonPath),
      '--remove-existing',
    ];
    if (pageless) args.push('--pageless');

    const result = execSync(
      `"${addLinksBin}" ${args.join(' ')}`,
      { stdio: ['pipe', 'pipe', 'pipe'], timeout: 30000 },
    );

    // Parse JSON result from stdout
    const resultStr = result.toString().trim();
    if (resultStr) {
      return JSON.parse(resultStr);
    }
    return { added: 0, removed: 0, skipped: 0 };
  } catch (e) {
    // Log stderr from the Swift tool if available
    if (e.stderr) {
      const stderr = e.stderr.toString().trim();
      if (stderr) console.warn('add-links stderr:', stderr);
    }
    throw e;
  } finally {
    try { unlinkSync(tmpJsonPath); } catch {}
  }
}

/**
 * Build a PDF document outline (bookmark tree) from heading data.
 *
 * Creates the /Outlines dictionary tree required by the PDF spec (ISO 32000).
 * Each heading becomes a bookmark item with a /Dest pointing to the heading's
 * position in the PDF. Heading levels map to outline depth:
 *   H1 = top-level, H2 = child of preceding H1, H3 = child of preceding H2, etc.
 *
 * Skipped levels (e.g., H1 then H3) are handled by treating the H3 as a child
 * of the nearest ancestor with a lower level — no synthetic intermediate nodes.
 *
 * Open/closed state: top-level items are expanded (positive Count), deeper items
 * are collapsed (negative Count). This matches the convention of most PDF tools.
 *
 * @param {PDFDocument} pdfDoc - The pdf-lib document to modify
 * @param {Array} headings - Array from getHeadingList(): [{id, raw, level, text}, ...]
 * @param {object} linkData - { anchors: {id: {x,y,w,h}} } from COLLECT_LINKS_JS (may be null for WebKit)
 * @param {object} opts - { pageless: boolean }
 * @returns {number} Number of outline items created
 */
function addPdfOutline(pdfDoc, headings, linkData, { pageless }) {
  if (!headings || headings.length === 0) return 0;

  const { PDFName, PDFNumber, PDFHexString, PDFNull } = require('pdf-lib');
  const ctx = pdfDoc.context;
  const pages = pdfDoc.getPages();

  if (pages.length === 0) return 0;

  const page0 = pages[0];
  const pageHeight = page0.getHeight();
  const pageRef = page0.ref;

  // For paged mode, we need additional geometry
  let pageCount, pageHeightPts, marginTopPts, marginBotPts, marginLeftPts;
  let contentHeightPts, contentWidthPts, xScale, yScale, cssPerPage;
  const scale = 0.75; // CSS px to PDF pts (72/96)

  if (!pageless && pages.length > 1) {
    pageCount = pages.length;
    pageHeightPts = pages[0].getHeight();
    const pageWidthPts = pages[0].getWidth();
    marginTopPts = 25.0 * 72.0 / 25.4;   // 70.87 pts
    marginBotPts = 25.0 * 72.0 / 25.4;
    marginLeftPts = 20.0 * 72.0 / 25.4;   // 56.69 pts
    contentHeightPts = pageHeightPts - marginTopPts - marginBotPts;
    contentWidthPts = pageWidthPts - 2.0 * marginLeftPts;
    const viewportWidth = (linkData && linkData.viewportWidth) || 1400;
    xScale = contentWidthPts / viewportWidth;
    yScale = xScale;
    cssPerPage = contentHeightPts / yScale;
  }

  /**
   * Map CSS Y coordinate to a PDF destination: { pageIndex, pdfY, pageRef }
   */
  function cssToDest(cssY) {
    if (pageless || pages.length === 1) {
      return { pageIndex: 0, pdfY: pageHeight - (cssY * scale), pageRef };
    }
    // Paged mode
    const pageIdx = Math.min(Math.floor(cssY / cssPerPage), pageCount - 1);
    const yWithinContent = cssY - pageIdx * cssPerPage;
    const pdfYFromTop = marginTopPts + yWithinContent * yScale;
    const pdfY = pageHeightPts - pdfYFromTop;
    return { pageIndex: pageIdx, pdfY, pageRef: pages[pageIdx].ref };
  }

  // Strip markdown inline formatting from heading text for clean bookmark titles.
  // The `raw` field from getHeadingList() may contain **bold**, *italic*, `code`, etc.
  function cleanTitle(raw) {
    return raw
      .replace(/\*\*(.+?)\*\*/g, '$1')   // **bold**
      .replace(/\*(.+?)\*/g, '$1')        // *italic*
      .replace(/__(.+?)__/g, '$1')        // __bold__
      .replace(/_(.+?)_/g, '$1')          // _italic_
      .replace(/~~(.+?)~~/g, '$1')        // ~~strike~~
      .replace(/`(.+?)`/g, '$1')          // `code`
      .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1') // [text](url)
      .trim();
  }

  // Build tree nodes from flat heading list
  const items = [];
  for (const h of headings) {
    const title = cleanTitle(h.raw);
    if (!title) continue; // skip empty headings

    // Look up Y position from anchor data (heading ID -> element position)
    let dest;
    if (linkData && linkData.anchors && linkData.anchors[h.id]) {
      dest = cssToDest(linkData.anchors[h.id].y);
    } else {
      // No position data (e.g., WebKit path) — fall back to page top
      dest = { pageIndex: 0, pdfY: pageHeight, pageRef };
    }

    // Create the outline item dictionary (Parent/First/Last/Prev/Next added later)
    const dict = ctx.obj({
      Title: PDFHexString.fromText(title),
      Dest: [dest.pageRef, PDFName.of('XYZ'), PDFNumber.of(0), PDFNumber.of(dest.pdfY), PDFNull],
    });
    const ref = ctx.register(dict);

    items.push({ level: h.level, title, dict, ref, children: [] });
  }

  if (items.length === 0) return 0;

  // Build hierarchical tree using a stack.
  // Root is a virtual node at level 0 that holds top-level items.
  const root = { level: 0, children: [] };
  const stack = [root];

  for (const item of items) {
    // Pop stack until we find a parent at a lower level than this item.
    // This handles skipped levels naturally: H1->H3 makes H3 a child of H1.
    while (stack.length > 1 && stack[stack.length - 1].level >= item.level) {
      stack.pop();
    }
    stack[stack.length - 1].children.push(item);
    stack.push(item);
  }

  // Wire up PDF references: Parent, First, Last, Prev, Next, Count
  function wireChildren(parent, parentRef, depth) {
    if (parent.children.length === 0) return 0;

    let totalDescendants = 0;
    for (let i = 0; i < parent.children.length; i++) {
      const child = parent.children[i];
      child.dict.set(PDFName.of('Parent'), parentRef);

      if (i > 0) {
        child.dict.set(PDFName.of('Prev'), parent.children[i - 1].ref);
      }
      if (i < parent.children.length - 1) {
        child.dict.set(PDFName.of('Next'), parent.children[i + 1].ref);
      }

      const childDescendants = wireChildren(child, child.ref, depth + 1);
      if (child.children.length > 0) {
        child.dict.set(PDFName.of('First'), child.children[0].ref);
        child.dict.set(PDFName.of('Last'), child.children[child.children.length - 1].ref);
        // Top-level items (depth 0) are open (positive Count).
        // Deeper items are collapsed (negative Count) for cleaner sidebar.
        const count = childDescendants;
        child.dict.set(PDFName.of('Count'), PDFNumber.of(depth < 1 ? count : -count));
      }

      totalDescendants += 1 + childDescendants;
    }
    return totalDescendants;
  }

  // Create outline root dictionary
  const outlineRootDict = ctx.obj({ Type: 'Outlines' });
  const outlineRootRef = ctx.register(outlineRootDict);

  const totalCount = wireChildren(root, outlineRootRef, 0);

  if (root.children.length > 0) {
    outlineRootDict.set(PDFName.of('First'), root.children[0].ref);
    outlineRootDict.set(PDFName.of('Last'), root.children[root.children.length - 1].ref);
    // Root Count = total visible items (all top-level + their open descendants)
    outlineRootDict.set(PDFName.of('Count'), PDFNumber.of(totalCount));
  }

  // Attach to document catalog
  pdfDoc.catalog.set(PDFName.of('Outlines'), outlineRootRef);
  // Set PageMode to UseOutlines so the sidebar opens automatically
  pdfDoc.catalog.set(PDFName.of('PageMode'), PDFName.of('UseOutlines'));

  return items.length;
}

/**
 * Post-process a PDF to inject metadata and document outline (bookmarks).
 *
 * Combines two operations into a single read-write cycle to avoid redundant I/O:
 * 1. PDF metadata from YAML front matter (title, author, subject, etc.)
 * 2. Document outline (bookmark sidebar) from heading hierarchy
 *
 * Metadata mapping:
 *   title       → pdfDoc.setTitle()
 *   author      → pdfDoc.setAuthor()
 *   subject|description → pdfDoc.setSubject()
 *   keywords|tags       → pdfDoc.setKeywords() (arrays joined with comma)
 *   date        → pdfDoc.setCreationDate() (ISO date string → Date object)
 *   creator     → pdfDoc.setCreator()  (default: "markdown-kit")
 *   producer    → pdfDoc.setProducer() (default: "markdown-kit + <engine>")
 *
 * @param {string} pdfPath - Path to the PDF file to modify in-place
 * @param {object} frontMatter - Parsed YAML front matter object (from gray-matter)
 * @param {string} engineName - "chromium" or "webkit", used for default producer
 * @param {Array} headings - Heading list from getHeadingList()
 * @param {object|null} linkData - Anchor position data from COLLECT_LINKS_JS (null for WebKit)
 * @param {object} pdfOpts - { pageless: boolean }
 * @returns {{ outlineCount: number }} Number of outline items created
 */
async function injectPdfMetadataAndOutline(pdfPath, frontMatter, engineName, headings, linkData, pdfOpts) {
  const hasMetadata = frontMatter && Object.keys(frontMatter).length > 0;
  const hasHeadings = headings && headings.length > 0;

  if (!hasMetadata && !hasHeadings) return { outlineCount: 0 };

  const pdfBytes = readFileSync(pdfPath);
  const pdfDoc = await PDFDocument.load(pdfBytes);

  // ── Metadata injection ──────────────────────────────────────────────────────
  if (hasMetadata) {
    if (frontMatter.title) pdfDoc.setTitle(String(frontMatter.title));
    if (frontMatter.author) pdfDoc.setAuthor(String(frontMatter.author));

    const subject = frontMatter.subject || frontMatter.description;
    if (subject) pdfDoc.setSubject(String(subject));

    const keywords = frontMatter.keywords || frontMatter.tags;
    if (keywords) {
      const kw = Array.isArray(keywords) ? keywords.join(', ') : String(keywords);
      pdfDoc.setKeywords(kw.split(',').map(s => s.trim()));
    }

    if (frontMatter.date) {
      const parsed = new Date(String(frontMatter.date));
      if (!isNaN(parsed.getTime())) pdfDoc.setCreationDate(parsed);
    }

    pdfDoc.setCreator(frontMatter.creator ? String(frontMatter.creator) : 'markdown-kit');
    pdfDoc.setProducer(
      frontMatter.producer
        ? String(frontMatter.producer)
        : `markdown-kit + ${engineName === 'webkit' ? 'WebKit' : 'Chromium'}`
    );
  } else {
    // Always set Creator/Producer even without front matter
    pdfDoc.setCreator('markdown-kit');
    pdfDoc.setProducer(`markdown-kit + ${engineName === 'webkit' ? 'WebKit' : 'Chromium'}`);
  }

  // ── Outline (bookmarks) injection ───────────────────────────────────────────
  let outlineCount = 0;
  if (hasHeadings) {
    outlineCount = addPdfOutline(pdfDoc, headings, linkData, pdfOpts);
  }

  const modifiedBytes = await pdfDoc.save();
  writeFileSync(pdfPath, Buffer.from(modifiedBytes));

  return { outlineCount };
}

// ── PDF generation ────────────────────────────────────────────────────────────

async function generatePdf() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--allow-file-access-from-files'],
  });

  const page = await browser.newPage();

  // Set viewport to match the desired width
  await page.setViewport({ width: pageWidth, height: 800 });

  // Write HTML to a temp file in the same directory as the markdown so relative
  // paths resolve correctly, then navigate to it via file:// (setContent runs
  // as about:blank which blocks file:// image loading)
  const tempHtmlPath = join(inputDir, `.pageless-tmp-${process.pid}.html`);
  writeFileSync(tempHtmlPath, fullHtml);

  let linkData = null;

  try {
    const tempHtmlUrl = 'file://' + tempHtmlPath.split('/').map(s => encodeURIComponent(s)).join('/');
    await page.goto(tempHtmlUrl, { waitUntil: 'networkidle0', timeout: 30000 });

    // Wait for images to load
    await page.evaluate(() => {
      const images = Array.from(document.querySelectorAll('img'));
      return Promise.all(images.map(img => {
        if (img.complete) return Promise.resolve();
        return new Promise((resolve) => {
          img.addEventListener('load', resolve);
          img.addEventListener('error', resolve);
        });
      }));
    });

    // Wait for fonts to load
    await page.evaluate(() => document.fonts.ready);

    // ── Mermaid diagram rendering ───────────────────────────────────────────
    // Convert ```mermaid code blocks into SVG diagrams in-browser.
    // marked outputs them as <pre><code class="hljs language-mermaid">...</code></pre>.
    // We rewrite those into <div class="mermaid"> containers, inject the
    // mermaid library, and let it render before measuring height.
    const hasMermaid = await page.evaluate(() =>
      document.querySelectorAll('pre > code.language-mermaid').length > 0
    );

    if (hasMermaid) {
      // Step 1: Replace <pre><code class="language-mermaid"> with <div class="mermaid">
      await page.evaluate(() => {
        const blocks = document.querySelectorAll('pre > code.language-mermaid');
        for (const code of blocks) {
          const pre = code.parentElement;
          const div = document.createElement('div');
          div.className = 'mermaid';
          div.textContent = code.textContent;
          pre.replaceWith(div);
        }
      });

      // Step 2: Inject mermaid.js bundle and render
      await page.addScriptTag({ path: mermaidBundlePath });
      await page.evaluate(async () => {
        window.mermaid.initialize({
          startOnLoad: false,
          theme: 'default',
          securityLevel: 'loose',
        });
        await window.mermaid.run({ querySelector: '.mermaid' });
      });
    }

    // ── Emulate print media for measurements ────────────────────────────────
    // Both pageless and paged modes need print emulation for accurate
    // element positions. Set a tall viewport so content flows without clipping.
    await page.emulateMediaType('print');
    await page.setViewport({ width: pageWidth, height: 50000 });
    await page.evaluate(() => void document.body.offsetHeight);

    // ── Collect internal link positions ────────────────────────────────────
    // Must happen after print emulation and layout reflow, before page.pdf().
    try {
      const linkJson = await page.evaluate(COLLECT_LINKS_JS);
      linkData = JSON.parse(linkJson);
    } catch (e) {
      // Non-fatal: if collection fails, we still generate the PDF without links
      console.warn('Warning: failed to collect internal link data:', e.message);
    }

    if (opts.pageless) {
      // ── Two-pass pageless approach ─────────────────────────────────────────
      // Pass 1: Measure where the last visible element ends.
      // Pass 2: Generate the PDF at that exact height.

      const contentBottom = await page.evaluate(() => {
        const children = Array.from(document.body.children);
        let lastBottom = 0;
        for (const el of children) {
          const rect = el.getBoundingClientRect();
          if (rect.height > 0 && rect.bottom > lastBottom) {
            lastBottom = rect.bottom;
          }
        }
        return Math.ceil(lastBottom);
      });

      const pdfHeight = contentBottom + 10;

      await page.pdf({
        path: outputPath,
        width: `${pageWidth}px`,
        height: `${pdfHeight}px`,
        printBackground: true,
        margin: { top: 0, right: 0, bottom: 0, left: 0 },
        preferCSSPageSize: false,
      });
    } else {
      // ── Normal paged mode (A4) ─────────────────────────────────────────────
      const { headerTemplate, footerTemplate, displayHeaderFooter } =
        resolveHeaderFooterTemplates(parsedFrontMatter, opts.theme);

      await page.pdf({
        path: outputPath,
        format: 'A4',
        printBackground: true,
        displayHeaderFooter,
        ...(displayHeaderFooter && { headerTemplate, footerTemplate }),
        margin: { top: '25mm', bottom: '25mm', left: '20mm', right: '20mm' },
      });
    }
  } finally {
    await browser.close().catch(() => {});
    try { unlinkSync(tempHtmlPath); } catch {}
  }

  // ── Post-process: add internal link annotations ────────────────────────────
  // Uses compiled Swift tool (add-links) with Apple PDFKit for guaranteed
  // compatibility with macOS Preview.app and iOS.
  if (linkData && linkData.links && linkData.links.length > 0) {
    try {
      const { added, removed, skipped } = addInternalLinkAnnotations(
        outputPath, linkData, { pageless: opts.pageless });
      if (added > 0 || removed > 0) {
        let msg = `Links:      ${added} internal link annotations`;
        if (removed > 0) msg += ` (replaced ${removed} Chromium-generated)`;
        if (skipped > 0) msg += ` (${skipped} targets not found)`;
        console.log(msg);
      }
    } catch (e) {
      // Non-fatal: if post-processing fails, keep the un-annotated PDF
      console.warn('Warning: failed to add internal links to PDF:', e.message);
    }
  }

  // ── Post-process: inject PDF metadata + document outline ─────────────────────
  try {
    const { outlineCount } = await injectPdfMetadataAndOutline(
      outputPath, parsedFrontMatter, 'chromium', parsedHeadings, linkData,
      { pageless: opts.pageless });
    if (outlineCount > 0) {
      console.log(`Bookmarks:  ${outlineCount} outline items from heading structure`);
    }
  } catch (e) {
    console.warn('Warning: failed to inject PDF metadata/outline:', e.message);
  }

  return { outputPath, pageWidth };
}

// ── WebKit PDF generation (macOS only) ────────────────────────────────────────

async function generatePdfWebKit() {
  // Locate the webkit-pdf binary next to this script
  const webkitBin = join(scriptDir, 'webkit-pdf');
  if (!existsSync(webkitBin)) {
    console.error(`Error: webkit-pdf binary not found at ${webkitBin}`);
    console.error('Compile it with: xcrun swiftc -o webkit-pdf webkit-pdf.swift -framework WebKit -framework AppKit');
    process.exit(1);
  }

  // Write the full HTML to a temp file (webkit-pdf loads from file path)
  const tempHtmlPath = join(tmpdir(), `pageless-webkit-${process.pid}.html`);
  writeFileSync(tempHtmlPath, fullHtml);

  try {
    const args = [tempHtmlPath, outputPath, '--width', String(pageWidth)];
    execSync(`"${webkitBin}" ${args.map(a => `"${a}"`).join(' ')}`, {
      stdio: ['pipe', 'inherit', 'inherit'],
      timeout: 30000,
    });
  } finally {
    try { unlinkSync(tempHtmlPath); } catch {}
  }

  // ── Post-process: inject PDF metadata + document outline ─────────────────────
  // WebKit path has no browser-collected anchor positions, so the outline
  // uses the Swift add-links tool's heading search for precise Y coordinates.
  // As a fallback, pdf-lib outline items point to page top when positions are unknown.
  try {
    // First, try using the Swift tool for precise bookmark placement
    const addLinksBin = join(scriptDir, 'add-links');
    if (existsSync(addLinksBin) && parsedHeadings.length > 0) {
      const headingsForSwift = parsedHeadings.map(h => ({
        title: h.raw
          .replace(/\*\*(.+?)\*\*/g, '$1')
          .replace(/\*(.+?)\*/g, '$1')
          .replace(/__(.+?)__/g, '$1')
          .replace(/_(.+?)_/g, '$1')
          .replace(/~~(.+?)~~/g, '$1')
          .replace(/`(.+?)`/g, '$1')
          .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
          .trim(),
        level: h.level,
        id: h.id,
      })).filter(h => h.title);

      if (headingsForSwift.length > 0) {
        const tmpHeadingsPath = join(tmpdir(), `pageless-headings-${process.pid}.json`);
        writeFileSync(tmpHeadingsPath, JSON.stringify(headingsForSwift));
        try {
          const result = execSync(
            `"${addLinksBin}" "${outputPath}" --add-bookmarks "${tmpHeadingsPath}"`,
            { stdio: ['pipe', 'pipe', 'pipe'], timeout: 30000 },
          );
          const resultStr = result.toString().trim();
          if (resultStr) {
            const parsed = JSON.parse(resultStr);
            if (parsed.bookmarks > 0) {
              console.log(`Bookmarks:  ${parsed.bookmarks} outline items from heading structure`);
            }
          }
        } catch (e) {
          if (e.stderr) {
            const stderr = e.stderr.toString().trim();
            if (stderr) console.warn('add-links bookmark stderr:', stderr);
          }
          // Fall through to pdf-lib fallback
        } finally {
          try { unlinkSync(tmpHeadingsPath); } catch {}
        }
      }
    }

    // Metadata injection (always via pdf-lib)
    const { outlineCount } = await injectPdfMetadataAndOutline(
      outputPath, parsedFrontMatter, 'webkit', [], null,
      { pageless: opts.pageless });
  } catch (e) {
    console.warn('Warning: failed to inject PDF metadata/outline:', e.message);
  }

  return { outputPath, pageWidth };
}

// ── Live preview server ───────────────────────────────────────────────────────

async function startServer() {
  const host = opts.host;
  let currentHtml = await buildHtml({ forServe: true });
  const sseClients = new Set();

  // MIME types for static file serving
  const MIME_TYPES = {
    '.html': 'text/html', '.css': 'text/css', '.js': 'application/javascript',
    '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml',
    '.webp': 'image/webp', '.ico': 'image/x-icon', '.woff': 'font/woff',
    '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.otf': 'font/otf',
    '.eot': 'application/vnd.ms-fontobject', '.pdf': 'application/pdf',
  };

  function getMime(filePath) {
    return MIME_TYPES[extname(filePath).toLowerCase()] || 'application/octet-stream';
  }

  function serveFile(res, filePath) {
    if (!existsSync(filePath)) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': getMime(filePath) });
    res.end(readFileSync(filePath));
  }

  const cssDir = dirname(stylesheetPath);

  const server = createServer((req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const pathname = decodeURIComponent(url.pathname);

    // GET / — serve the rendered HTML
    if (pathname === '/') {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(currentHtml);
      return;
    }

    // GET /events — SSE endpoint for live-reload
    if (pathname === '/events') {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      res.write('data: connected\n\n');
      sseClients.add(res);
      req.on('close', () => sseClients.delete(res));
      return;
    }

    // GET /mermaid.min.js — serve the mermaid browser bundle
    if (pathname === '/mermaid.min.js') {
      serveFile(res, mermaidBundlePath);
      return;
    }

    // GET /katex-fonts/* — serve KaTeX font files
    if (pathname.startsWith('/katex-fonts/')) {
      const fontFile = pathname.slice('/katex-fonts/'.length);
      serveFile(res, join(katexFontsDir, fontFile));
      return;
    }

    // GET /theme-assets/* — serve theme CSS assets (fonts, images)
    if (pathname.startsWith('/theme-assets/')) {
      const assetPath = pathname.slice('/theme-assets/'.length);
      serveFile(res, resolve(cssDir, assetPath));
      return;
    }

    // GET /static/* — serve files relative to the markdown's directory
    if (pathname.startsWith('/static/')) {
      const relPath = pathname.slice('/static/'.length);
      serveFile(res, resolve(inputDir, relPath));
      return;
    }

    res.writeHead(404);
    res.end('Not found');
  });

  // Ephemeral port: let the OS assign an available port
  server.listen(0, host, () => {
    const { port } = server.address();
    console.log(`  Local:   http://localhost:${port}`);
    if (host === '0.0.0.0') {
      // Find the local network IP for device access
      const nets = require('os').networkInterfaces();
      const localIp = Object.values(nets).flat()
        .find(n => n.family === 'IPv4' && !n.internal)?.address;
      if (localIp) {
        console.log(`  Network: http://${localIp}:${port}`);
      }
    } else {
      console.log('  Network: use --host 0.0.0.0 to expose');
    }
    console.log('Watching for changes... (Ctrl+C to stop)');
  });

  // File watcher with debounce
  let debounceTimer = null;
  const watchedPaths = [inputPath];
  if (existsSync(stylesheetPath)) watchedPaths.push(stylesheetPath);

  const watchers = watchedPaths.map(p => {
    return fsWatch(p, () => {
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(async () => { // 500ms debounce for iCloud Drive's noisy fs events
        try {
          currentHtml = await buildHtml({ forServe: true });
          for (const client of sseClients) {
            client.write('event: reload\ndata: refresh\n\n');
          }
          const now = new Date().toLocaleTimeString();
          console.log(`[${now}] Rebuilt — ${sseClients.size} client(s) notified`);
        } catch (e) {
          console.error(`[rebuild error] ${e.message}`);
        }
      }, 500);
    });
  });

  // Clean shutdown on Ctrl+C
  function shutdown() {
    console.log('\nShutting down...');
    for (const w of watchers) w.close();
    for (const client of sseClients) client.end();
    server.close(() => process.exit(0));
    // Force exit after 2s if server hangs
    setTimeout(() => process.exit(0), 2000);
  }
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

// ── Main ──────────────────────────────────────────────────────────────────────

if (opts.serve) {
  // Serve mode: live preview with auto-reload
  console.log(`Input:      ${inputPath}`);
  console.log(`Stylesheet: ${stylesheetPath}`);
  console.log(`Width:      ${pageWidth}px`);
  startServer();
} else if (opts.html) {
  // HTML output mode: write self-contained responsive HTML file
  console.log(`Input:      ${inputPath}`);
  console.log(`Stylesheet: ${stylesheetPath}`);
  console.log(`Width:      ${pageWidth}px`);
  const htmlContent = await buildHtml({ forServe: true });
  writeFileSync(outputPath, htmlContent);
  const sizeKB = (readFileSync(outputPath).length / 1024).toFixed(1);
  console.log(`Output:     ${outputPath}`);
  console.log(`Format:     HTML (self-contained, responsive)`);
  console.log(`Size:       ${sizeKB} KB`);
} else {
  // PDF generation mode
  console.log(`Input:      ${inputPath}`);
  console.log(`Stylesheet: ${stylesheetPath}`);
  console.log(`Width:      ${pageWidth}px`);
  console.log(`Engine:     ${engine}`);

  const result = engine === 'webkit'
    ? await generatePdfWebKit()
    : await generatePdf();

  const stats = readFileSync(result.outputPath);
  const sizeKB = (stats.length / 1024).toFixed(1);

  console.log(`Output:     ${result.outputPath}`);
  console.log(`Mode:       ${opts.pageless ? 'pageless' : 'paged (A4)'}`);
  if (!opts.pageless) {
    const hf = resolveHeaderFooterTemplates(parsedFrontMatter, opts.theme);
    if (hf.displayHeaderFooter) {
      console.log(`Footer:     page numbers enabled`);
    }
  }
  console.log(`Engine:     ${engine}`);
  console.log(`Size:       ${sizeKB} KB`);
}
