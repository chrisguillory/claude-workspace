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
//   Emoji:           We convert :shortcode: to Unicode glyphs via gemoji (GitHub)
//                    + Slack/Discord aliases for cross-platform shortcode support.
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
import { execSync, execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdirSync, unlinkSync, copyFileSync, statSync, watch as fsWatch } from 'node:fs';
import { resolve, dirname, basename, extname, join, relative } from 'node:path';
import { homedir, tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'node:http';

// ── Argument parsing ──────────────────────────────────────────────────────────

// Preprocess argv: --launch accepts an optional browser name (string flag).
// Node's parseArgs requires a value for string types, so inject 'default'
// when --launch is bare (followed by another flag or is the last arg).
const rawArgs = process.argv.slice(2);
for (let i = 0; i < rawArgs.length; i++) {
  if (rawArgs[i] === '--launch' && (i + 1 >= rawArgs.length || rawArgs[i + 1].startsWith('-'))) {
    rawArgs.splice(i + 1, 0, 'default');
  }
}

// Hardcoded defaults — config file and CLI flags override these.
// Boolean defaults are NOT set in parseArgs (which returns undefined for absent flags),
// enabling three-state detection: undefined=not set, true=--flag, false=--no-flag.
const DEFAULTS = {
  theme: 'pixyll',
  width: '1400',
  engine: 'chromium',
  'front-matter': 'strip',
  host: 'localhost',
  'toc-nav': '',
  html: false,
  pageless: true,
  'page-numbers': true,
  serve: false,
  mobile: false,
  'rich-highlighting': false,
  'macos-spoken-content': false,
  'embed-images': false,
  'copy-assets': false,
  'absolute-paths': false,
  'accept-broken-images': false,
  'secret-gist': false,
  'show-timestamp': false,
  'show-filepath': false,
  help: false,
};

const { values: cliOpts, positionals } = parseArgs({
  args: rawArgs,
  allowPositionals: true,
  allowNegative: true,
  options: {
    stylesheet: { type: 'string', short: 's' },
    theme:      { type: 'string', short: 't' },
    width:      { type: 'string', short: 'w' },
    engine:     { type: 'string', short: 'e' },
    'front-matter': { type: 'string' },
    html:       { type: 'boolean' },
    pageless:   { type: 'boolean' },
    'page-numbers': { type: 'boolean' },
    header:     { type: 'string' },
    footer:     { type: 'string' },
    serve:      { type: 'boolean' },
    host:       { type: 'string' },
    output:     { type: 'string', short: 'o' },
    mobile:     { type: 'boolean', short: 'm' },
    'rich-highlighting': { type: 'boolean' },
    'toc-nav':  { type: 'string' },
    'macos-spoken-content': { type: 'boolean' },
    'embed-images': { type: 'boolean' },
    'copy-assets': { type: 'boolean' },
    'absolute-paths': { type: 'boolean' },
    'accept-broken-images': { type: 'boolean' },
    'secret-gist': { type: 'boolean' },
    'gist-id':  { type: 'string' },
    'show-timestamp': { type: 'boolean' },
    'show-filepath': { type: 'boolean' },
    launch:     { type: 'string' },
    help:       { type: 'boolean', short: 'h' },
  },
});

// Help check uses cliOpts directly (before config merge — help is always CLI-driven)
if (cliOpts.help || positionals.length === 0) {
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
  --toc-nav <features>      TOC navigation: inject, backlinks, smooth, float, all (comma-separated)
  --macos-spoken-content    TTS-safe code blocks (fullwidth angle brackets for macOS Spoken Content)
  --embed-images            Embed images as base64 data URIs
  --copy-assets             Copy local images alongside output HTML
  --absolute-paths          Rewrite image paths to absolute (local machine only)
  --accept-broken-images    Proceed when local images won't resolve
  --secret-gist             Upload HTML to secret GitHub gist (implies --html)
  --gist-id <id|url>        Update existing gist instead of creating new
  --launch [browser]        Open browser in serve/gist mode (safari, chrome, chromium, firefox)

Output metadata:
  --show-timestamp          Show build timestamp in output
  --show-filepath           Show source file path in output
  --no-show-timestamp       Disable (override config)
  --no-show-filepath        Disable (override config)

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
  process.exit(cliOpts.help ? 0 : 1);
}

// ── Config file loading + three-layer merge ──────────────────────────────────
// Precedence: CLI flags > config file > hardcoded DEFAULTS
// Config file: ~/.claude-workspace/tools/markdown-kit/config.yaml
// Loaded BEFORE npm deps are installed (can't use js-yaml here), so we parse
// the simple flat key:value YAML subset inline. Supports comments (#) and
// boolean/string/number values.

const CONFIG_DIR = join(homedir(), '.claude-workspace', 'tools', 'markdown-kit');

function loadUserConfig() {
  const configPath = join(CONFIG_DIR, 'config.yaml');
  if (!existsSync(configPath)) return {};
  try {
    const config = {};
    for (const line of readFileSync(configPath, 'utf-8').split('\n')) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const colonIdx = trimmed.indexOf(':');
      if (colonIdx === -1) continue;
      const key = trimmed.slice(0, colonIdx).trim();
      let val = trimmed.slice(colonIdx + 1).trim();
      // Strip surrounding quotes (handle YAML quoted strings)
      if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
        val = val.slice(1, -1);
      } else {
        // Strip inline comments only for unquoted values (quoted values may contain #)
        const commentIdx = val.indexOf(' #');
        if (commentIdx !== -1) val = val.slice(0, commentIdx).trim();
      }
      // Type coercion
      if (val === 'true') config[key] = true;
      else if (val === 'false') config[key] = false;
      else if (/^\d+$/.test(val)) config[key] = val;  // keep as string (width, etc.)
      else config[key] = val;
    }
    return config;
  } catch (e) {
    console.warn(`Warning: failed to parse config: ${e.message}`);
    return {};
  }
}

const userConfig = loadUserConfig();
const opts = {};

// Merge defaulted flags: CLI > config > DEFAULTS
for (const key of Object.keys(DEFAULTS)) {
  if (cliOpts[key] !== undefined) opts[key] = cliOpts[key];
  else if (key in userConfig) opts[key] = userConfig[key];
  else opts[key] = DEFAULTS[key];
}

// Pass through non-defaulted string flags (no hardcoded default — undefined if absent)
for (const key of ['stylesheet', 'header', 'footer', 'output', 'gist-id', 'launch']) {
  if (cliOpts[key] !== undefined) opts[key] = cliOpts[key];
  else if (key in userConfig) opts[key] = userConfig[key];
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
// --mobile preset: override width to 430px (iPhone) unless --width was explicitly set on CLI
if (opts.mobile && cliOpts.width === undefined) {
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

// Parse --toc-nav features into a Set
const validTocFeatures = new Set(['inject', 'backlinks', 'smooth', 'float']);
const tocNavFeatures = new Set(
  opts['toc-nav'] === 'all' ? ['inject', 'backlinks', 'smooth', 'float']
    : opts['toc-nav'].split(',').map(s => s.trim()).filter(Boolean)
);
for (const f of tocNavFeatures) {
  if (!validTocFeatures.has(f)) {
    console.error(`Error: unknown --toc-nav feature '${f}'. Valid: inject, backlinks, smooth, float, all`);
    process.exit(1);
  }
}

// --secret-gist implies --html
if (opts['secret-gist']) {
  opts.html = true;
}

// --secret-gist and --serve are mutually exclusive
if (opts['secret-gist'] && opts.serve) {
  console.error('Error: --serve and --secret-gist are mutually exclusive.');
  process.exit(1);
}

// --launch requires --serve or --secret-gist
// Only error if explicitly passed via CLI; silently ignore if from config (config may set launch
// globally but not every invocation uses serve/gist mode)
if (opts.launch !== undefined && !opts.serve && !opts['secret-gist']) {
  if (cliOpts.launch !== undefined) {
    console.error('Error: --launch requires --serve or --secret-gist.');
    process.exit(1);
  }
  delete opts.launch;  // config-sourced launch, not applicable for this mode
}

// --embed-images validation
if (opts['embed-images'] && !opts.html && !opts.serve) {
  console.error('Error: --embed-images only applies to HTML or serve output. PDF embeds images via the browser engine.');
  process.exit(1);
}

// Image resolution flags are mutually exclusive
const imageFlags = ['embed-images', 'copy-assets', 'absolute-paths', 'accept-broken-images'].filter(f => opts[f]);
if (imageFlags.length > 1) {
  console.error(`Error: --${imageFlags.join(', --')} are mutually exclusive. Choose one.`);
  process.exit(1);
}

// --copy-assets validation
if (opts['copy-assets']) {
  if (opts.serve) { console.error('Error: --copy-assets has no effect in serve mode.'); process.exit(1); }
  if (opts['secret-gist']) { console.error('Error: --copy-assets cannot be used with --secret-gist. Use --embed-images.'); process.exit(1); }
  if (!opts.html) { console.error('Error: --copy-assets only applies to HTML output.'); process.exit(1); }
}

// --absolute-paths validation
if (opts['absolute-paths']) {
  if (opts.serve) { console.error('Error: --absolute-paths has no effect in serve mode.'); process.exit(1); }
  if (opts['secret-gist']) { console.error('Error: --absolute-paths cannot be used with --secret-gist. Use --embed-images.'); process.exit(1); }
  if (!opts.html) { console.error('Error: --absolute-paths only applies to HTML output.'); process.exit(1); }
}

// --accept-broken-images validation
if (opts['accept-broken-images']) {
  if (!opts.html && !opts.serve) { console.error('Error: --accept-broken-images only applies to HTML or serve output.'); process.exit(1); }
}

// --gist-id requires --secret-gist
if (opts['gist-id'] && !opts['secret-gist']) {
  console.warn('Warning: --gist-id requires --secret-gist. Ignoring --gist-id.');
}

// --secret-gist requires gh CLI authenticated
if (opts['secret-gist']) {
  try {
    execSync('gh auth status', { stdio: ['pipe', 'pipe', 'pipe'] });
  } catch {
    console.error('Error: --secret-gist requires the GitHub CLI (gh) to be installed and authenticated.');
    console.error('  Install: brew install gh');
    console.error('  Login:   gh auth login');
    process.exit(1);
  }
}

// --launch with empty string should fail fast
if (opts.launch !== undefined && opts.launch === '') {
  console.error('Error: --launch requires a browser name or omit value for system default.');
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
  gemoji: '^8',
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
const { nameToEmoji } = require('gemoji');
const markedFootnote = require('marked-footnote');
const markedAlert = require('marked-alert');
const grayMatter = require('gray-matter');
const { PDFDocument } = require('pdf-lib');

// Resilient frontmatter parsing — handles malformed YAML (e.g., Claude Code agent files
// with unquoted colons, angle brackets, escape sequences in description fields).
// Mirrors Claude Code's own two-pass strategy from src/utils/frontmatterParser.ts.
const YAML_SPECIAL_CHARS = /[{}[\]*&#!|>%@`]|: /;

function quoteProblematicValues(yamlText) {
  return yamlText.split('\n').map(line => {
    const match = line.match(/^([a-zA-Z_-]+):\s+(.+)$/);
    if (!match) return line;
    const [, key, value] = match;
    if (!key || !value) return line;
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) return line;
    if (YAML_SPECIAL_CHARS.test(value)) {
      const escaped = value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
      return `${key}: "${escaped}"`;
    }
    return line;
  }).join('\n');
}

const jsYaml = require('js-yaml');
const matterOptions = {
  engines: {
    yaml: {
      parse: (str) => {
        try { return jsYaml.load(str); }
        catch { return jsYaml.load(quoteProblematicValues(str)); }
      },
      stringify: (obj) => jsYaml.dump(obj),
    },
  },
};
function matter(source) { return grayMatter(source, matterOptions); }

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
// Primary: gemoji (GitHub's official dictionary, 1913 shortcodes)
// Supplementary: Slack/Discord aliases for shortcodes that differ from GitHub convention
// (e.g., :robot_face: → 🤖 where GitHub uses :robot:)
const slackAliasesData = JSON.parse(readFileSync(join(scriptDir, 'slack-emoji-aliases.json'), 'utf-8'));
const emojiMap = { ...nameToEmoji, ...slackAliasesData.aliases };
marked.use(markedEmoji({ emojis: emojiMap, renderer: (token) => token.emoji }));

// Staleness nudge (6-month threshold) — emoji-datasource tracks annual Unicode releases
if (slackAliasesData._generated_at && process.env.MARKDOWN_KIT_IGNORE_STALE_EMOJI !== '1') {
  const monthsOld = (Date.now() - new Date(slackAliasesData._generated_at)) / (1000 * 60 * 60 * 24 * 30);
  if (monthsOld >= 6) {
    console.warn(`markdown-kit: emoji aliases are ${Math.floor(monthsOld)} months old. Run 'node tools/markdown-kit/scripts/regenerate-emoji-aliases.js' to refresh.`);
  }
}

// ── Reusable pipeline functions ────────────────────────────────────────────────

function generateTocHtml(headings) {
  if (!headings || headings.length === 0) return '';
  const items = headings.map(({ id, raw, level }) => {
    const indent = (level - 1) * 20;
    return `<li id="toc-${id}" style="margin-left:${indent}px"><a href="#${id}">${raw}</a></li>`;
  });
  return `<div class="md-toc" id="md-toc-top"><ul>\n${items.join('\n')}\n</ul></div>`;
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

// MIME type lookup for image embedding
const IMAGE_MIME = {
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.svg': 'image/svg+xml', '.webp': 'image/webp',
  '.ico': 'image/x-icon', '.avif': 'image/avif',
};

// Detect MIME type from binary magic bytes (fallback when URL has no extension)
function detectMimeFromBytes(buf) {
  if (buf[0] === 0x89 && buf[1] === 0x50 && buf[2] === 0x4E && buf[3] === 0x47) return 'image/png';
  if (buf[0] === 0xFF && buf[1] === 0xD8 && buf[2] === 0xFF) return 'image/jpeg';
  if (buf[0] === 0x47 && buf[1] === 0x49 && buf[2] === 0x46) return 'image/gif';
  if (buf[0] === 0x52 && buf[1] === 0x49 && buf[2] === 0x46 && buf[3] === 0x46) return 'image/webp';
  if (buf[0] === 0x3C) return 'image/svg+xml'; // starts with <
  return null;
}

// Embed images as base64 data URIs (--embed-images)
// Handles local files, remote URLs, and GitHub-authenticated URLs.
// Deduplicates repeated references. Skips images exceeding 5MB.
// Cache is module-scoped to persist across serve-mode rebuilds.
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const imageEmbedCache = new Map();

async function embedImages(html, baseDir) {
  let count = 0, totalOrigBytes = 0;
  const imgRegex = /(<img\s[^>]*src=")([^"]+)(")/gi;

  // Collect all matches first (can't use async in replace callback)
  const matches = [];
  let m;
  while ((m = imgRegex.exec(html)) !== null) {
    matches.push({ full: m[0], pre: m[1], src: m[2], post: m[3], index: m.index });
  }

  // Process each match, building replacements
  const replacements = [];
  for (const { full, pre, src, post } of matches) {
    if (/^data:/i.test(src)) { replacements.push(full); continue; }
    if (/^\//i.test(src)) { replacements.push(full); continue; }  // serve-mode /static/ paths
    if (imageEmbedCache.has(src)) { replacements.push(`${pre}${imageEmbedCache.get(src)}${post}`); count++; continue; }

    try {
      let data, mime;

      if (/^https?:\/\//i.test(src)) {
        // Remote URL — check if GitHub-hosted (needs auth)
        const isGitHub = /github\.com|githubusercontent\.com|user-attachments/i.test(src);
        if (isGitHub) {
          // Use gh CLI for authenticated GitHub fetches
          const result = execSync(`gh api "${src}" --method GET`, {
            stdio: ['pipe', 'pipe', 'pipe'], timeout: 15000, maxBuffer: MAX_IMAGE_BYTES,
            encoding: 'buffer',
          });
          data = result;
        } else {
          // Fetch via Node fetch with timeout
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 10000);
          try {
            const resp = await fetch(src, { signal: controller.signal });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            data = Buffer.from(await resp.arrayBuffer());
          } finally { clearTimeout(timer); }
        }
        const ext = extname(new URL(src).pathname).toLowerCase();
        mime = IMAGE_MIME[ext] || detectMimeFromBytes(data) || 'application/octet-stream';
      } else {
        // Local file — resolve against baseDir
        const absPath = resolve(baseDir, src);
        if (!existsSync(absPath)) { console.warn(`Warning: image not found, skipping: ${src}`); replacements.push(full); continue; }
        data = readFileSync(absPath);
        mime = IMAGE_MIME[extname(absPath).toLowerCase()] || detectMimeFromBytes(data) || 'application/octet-stream';
      }

      if (data.length > MAX_IMAGE_BYTES) {
        const sizeMB = (data.length / 1048576).toFixed(1);
        console.warn(`Warning: image exceeds 5 MB (${sizeMB} MB), skipping: ${src}`);
        replacements.push(full);
        continue;
      }

      totalOrigBytes += data.length;
      const dataUri = `data:${mime};base64,${data.toString('base64')}`;
      imageEmbedCache.set(src, dataUri);
      replacements.push(`${pre}${dataUri}${post}`);
      count++;
    } catch (e) {
      console.warn(`Warning: failed to embed image, skipping: ${src} (${e.message})`);
      replacements.push(full);
    }
  }

  // Rebuild HTML with replacements
  let result = html;
  for (let i = matches.length - 1; i >= 0; i--) {
    result = result.slice(0, matches[i].index) + replacements[i] + result.slice(matches[i].index + matches[i].full.length);
  }

  return { html: result, count, totalOrigBytes };
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

// Detect local image references in HTML (not data:, not http/https)
function detectLocalImages(html) {
  const localImages = [];
  const imgRegex = /<img\s[^>]*src="([^"]+)"/gi;
  let m;
  while ((m = imgRegex.exec(html)) !== null) {
    const src = m[1];
    if (!/^(data:|https?:\/\/)/i.test(src)) {
      localImages.push(src);
    }
  }
  return localImages;
}

// Rewrite local image paths to file:// absolute paths (--absolute-paths)
function rewriteToAbsolutePaths(html, baseDir) {
  return html.replace(
    /(<img\s[^>]*src=")([^"]+)(")/gi,
    (match, pre, src, post) => {
      if (/^(data:|https?:\/\/|file:)/i.test(src)) return match;
      const absPath = resolve(baseDir, src);
      const encoded = absPath.split('/').map(s => encodeURIComponent(s)).join('/');
      return `${pre}file://${encoded}${post}`;
    }
  );
}

// Copy local images alongside output HTML (--copy-assets)
// Preserves relative directory structure inside {stem}_files/
function copyLocalAssets(html, baseDir, outputPath) {
  const outputDir = dirname(outputPath);
  const stem = basename(outputPath, extname(outputPath));
  const assetsDir = join(outputDir, `${stem}_files`);
  const imgRegex = /<img\s[^>]*src="([^"]+)"/gi;

  // Collect all local image references
  const matches = [];
  let m;
  while ((m = imgRegex.exec(html)) !== null) {
    const src = m[1];
    if (!/^(data:|https?:\/\/|file:)/i.test(src)) {
      matches.push({ full: m[0], src });
    }
  }

  if (matches.length === 0) return { html, count: 0, assetsDir: null };

  const copied = new Set(); // dedup: same src referenced multiple times
  let count = 0;

  for (const { full, src } of matches) {
    const absPath = resolve(baseDir, src);
    if (!existsSync(absPath)) {
      console.warn(`Warning: image not found, skipping copy: ${src}`);
      continue;
    }

    // Preserve relative path structure: ./images/foo.png → {stem}_files/images/foo.png
    // For parent refs (../assets/logo.svg), use just the filename to avoid escaping the assets dir
    const relPath = relative(baseDir, absPath);
    const destRel = relPath.startsWith('..') ? basename(absPath) : relPath;
    const destPath = join(assetsDir, destRel);

    if (!copied.has(absPath)) {
      mkdirSync(dirname(destPath), { recursive: true });
      copyFileSync(absPath, destPath);
      copied.add(absPath);
      count++;
    }

    const newSrc = `${stem}_files/${destRel}`;
    html = html.split(full).join(full.replace(src, newSrc));
  }

  return { html, count, assetsDir };
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
function loadKatexCss(forServe, forStandalone = false) {
  if (!existsSync(katexCssPath)) return '';
  let css = readFileSync(katexCssPath, 'utf-8');
  if (forServe) {
    // Route font URLs through our HTTP server
    css = css.replace(/url\(fonts\//g, 'url(/katex-fonts/');
  } else if (forStandalone) {
    // Use jsDelivr CDN for KaTeX fonts — semver range pins to latest 0.16.x
    css = css.replace(
      /url\(fonts\//g,
      'url(https://cdn.jsdelivr.net/npm/katex@0.16/dist/fonts/'
    );
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
function loadThemeCss(forServe, forStandalone = false) {
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
  } else if (forStandalone) {
    // Replace local .woff font URLs with jsDelivr CDN serving fontsource packages.
    // fontsource provides stable, versioned URLs for Google Fonts (no hash churn).
    // Pattern: @fontsource/<family>/files/<family>-latin-<weight>-<style>.<format>
    const FONT_CDN_MAP = {
      'merriweather-v19-latin-300.woff':        'https://cdn.jsdelivr.net/npm/@fontsource/merriweather/files/merriweather-latin-300-normal.woff',
      'merriweather-v19-latin-700.woff':        'https://cdn.jsdelivr.net/npm/@fontsource/merriweather/files/merriweather-latin-700-normal.woff',
      'merriweather-v19-latin-300italic.woff':  'https://cdn.jsdelivr.net/npm/@fontsource/merriweather/files/merriweather-latin-300-italic.woff',
      'merriweather-v19-latin-700italic.woff':  'https://cdn.jsdelivr.net/npm/@fontsource/merriweather/files/merriweather-latin-700-italic.woff',
      'lato-v14-latin-300.woff':                'https://cdn.jsdelivr.net/npm/@fontsource/lato/files/lato-latin-300-normal.woff',
      'lato-v14-latin-900.woff':                'https://cdn.jsdelivr.net/npm/@fontsource/lato/files/lato-latin-900-normal.woff',
      'lato-v14-latin-300italic.woff':          'https://cdn.jsdelivr.net/npm/@fontsource/lato/files/lato-latin-300-italic.woff',
      'lato-v14-latin-900italic.woff':          'https://cdn.jsdelivr.net/npm/@fontsource/lato/files/lato-latin-900-italic.woff',
    };
    css = css.replace(
      /url\(['"]?(\.\/?([^'")]+))['"]?\)/g,
      (match, relPath, filename) => {
        const cdnUrl = FONT_CDN_MAP[filename];
        return cdnUrl ? `url('${cdnUrl}')` : match;
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

  // Resolve header content: CLI > front matter > timestamp (if enabled) > none
  let headerContent = opts.header || null;
  if (!headerContent && frontMatter && frontMatter.header) {
    headerContent = String(frontMatter.header);
  }
  if (!headerContent && opts['show-timestamp']) {
    headerContent = `Generated ${new Date().toLocaleString()}`;
  }

  // Resolve footer content: CLI > front matter > filepath+pages > default page numbers
  let footerContent = opts.footer || null;
  if (!footerContent && frontMatter && frontMatter.footer) {
    footerContent = String(frontMatter.footer);
  }
  if (!footerContent && opts['show-filepath'] && pageNumbersEnabled) {
    const dp = inputPath.replace(homedir(), '~');
    footerContent = `${dp} \u00B7 Page {{pageNumber}} of {{totalPages}}`;
  } else if (!footerContent && opts['show-filepath']) {
    footerContent = inputPath.replace(homedir(), '~');
  } else if (!footerContent && pageNumbersEnabled) {
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
async function buildHtml({ forServe = false, forStandalone = false } = {}) {
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

  // TOC generation: replace [toc] placeholders and track presence
  let processedBody = htmlBody;
  const tocHtml = generateTocHtml(getHeadingList());
  let hasToc = false;
  processedBody = processedBody.replace(/<p>\s*\[toc\]\s*<\/p>/gi, () => {
    hasToc = true;
    return tocHtml;
  });

  // --toc-nav inject: generate TOC if document doesn't have [toc]
  if (!hasToc && tocNavFeatures.has('inject')) {
    if (!tocHtml) {
      console.error('Error: --toc-nav inject failed — document has no headings.');
      process.exit(1);
    }
    processedBody = processedBody.replace(/(<h[1-6]\s)/i, tocHtml + '\n$1');
    hasToc = true;
  }

  // Fail-fast: backlinks/float require a TOC
  if (!hasToc && (tocNavFeatures.has('backlinks') || tocNavFeatures.has('float'))) {
    console.error('Error: --toc-nav backlinks/float requires [toc] in document (or use inject). No TOC found.');
    process.exit(1);
  }

  // --toc-nav backlinks: wrap heading text in links back to their TOC entry
  if (hasToc && tocNavFeatures.has('backlinks')) {
    processedBody = processedBody.replace(
      /<(h[1-6])\s+id="([^"]+)">([\s\S]*?)<\/\1>/gi,
      (match, tag, id, content) =>
        `<${tag} id="${id}"><a class="heading-backlink" href="#toc-${id}">${content}</a></${tag}>`
    );
  }

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
  // forServe: rewrite to /static/ for live server
  // forStandalone: leave as-is (relative paths from markdown; embedImages() resolves them)
  // default (PDF): rewrite to file:// for Puppeteer/WebKit
  let resolvedBody;
  if (forServe) {
    resolvedBody = resolveImagePathsForServe(processedBody);
  } else if (forStandalone) {
    resolvedBody = processedBody;
  } else {
    resolvedBody = resolveImagePaths(processedBody, inputDir);
  }

  // Embed images as base64 data URIs (--embed-images, standalone + serve mode)
  if (opts['embed-images'] && (forStandalone || forServe)) {
    const result = await embedImages(resolvedBody, inputDir);
    resolvedBody = result.html;
    if (result.count > 0) {
      const origMB = (result.totalOrigBytes / 1048576).toFixed(1);
      console.error(`Embedded:   ${result.count} image${result.count !== 1 ? 's' : ''} (${origMB} MB)`);
    } else if (!forServe) {
      // Only warn in standalone mode — in serve mode, local images are served via /static/
      // and are expected to be skipped by embedImages()
      console.warn('Warning: --embed-images passed but no images found to embed.');
    }
  }

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

  // --macos-spoken-content: replace angle brackets in code blocks with fullwidth Unicode
  // (U+FF1C/U+FF1E) to prevent macOS Spoken Content from stripping content between < and >.
  // Known issue: Chrome uses a clipboard-based TTS pathway (evidenced by Maccy issue #968)
  // that applies NFKC normalization, converting fullwidth back to ASCII. Works in Safari
  // and Chromium. No fix available — Chrome's accessibility text extraction is proprietary.
  if (opts['macos-spoken-content']) {
    // Skip diagram/chart languages — these are parsed by Mermaid/Graphviz/Vega-Lite, not displayed as text
    const diagramLangs = /language-(?:mermaid|dot|graphviz|vega-lite|chart)/i;
    resolvedBody = resolvedBody.replace(
      /<pre><code([^>]*)>([\s\S]*?)<\/code><\/pre>/gi,
      (match, attrs, inner) => {
        if (diagramLangs.test(attrs)) return match; // leave diagram blocks untouched
        let result = inner;
        // Replace HTML entities with styled fullwidth chars
        result = result.replace(/&lt;/g, '<span class="tts-bracket">\uFF1C</span>');
        result = result.replace(/&gt;/g, '<span class="tts-bracket">\uFF1E</span>');
        // sr-only pronunciation for comparison operators (space-delimited context)
        result = result.replace(
          / <span class="tts-bracket">\uFF1E<\/span> /g,
          ' <span class="tts-bracket">\uFF1E</span><span class="sr-only"> greater than </span> '
        );
        result = result.replace(
          / <span class="tts-bracket">\uFF1C<\/span> /g,
          ' <span class="tts-bracket">\uFF1C</span><span class="sr-only"> less than </span> '
        );
        // sr-only pronunciation for arrow functions (=＞)
        result = result.replace(
          /=<span class="tts-bracket">\uFF1E<\/span>/g,
          '=<span class="tts-bracket">\uFF1E</span><span class="sr-only"> arrow </span>'
        );
        return `<pre><code${attrs}>${result}</code></pre>`;
      }
    );

    // Also process inline <code> elements (backtick code, not inside <pre>).
    // Inline <code> from marked has no class attribute; fenced blocks have class="hljs ...".
    // Fenced blocks are already processed above, so their &lt;/&gt; are gone — no double-processing.
    resolvedBody = resolvedBody.replace(
      /<code>([\s\S]*?)<\/code>/gi,
      (match, inner) => {
        if (!inner.includes('&lt;') && !inner.includes('&gt;')) return match;
        let result = inner;
        result = result.replace(/&lt;/g, '<span class="tts-bracket">\uFF1C</span>');
        result = result.replace(/&gt;/g, '<span class="tts-bracket">\uFF1E</span>');
        return `<code>${result}</code>`;
      }
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
  const themeCss = loadThemeCss(forServe, forStandalone);
  const kCss = loadKatexCss(forServe, forStandalone);

  // Serve mode extras: client-side mermaid + SSE live-reload
  // Floating "↑ TOC" button — serve + HTML export only (needs JS for IntersectionObserver)
  const showFloat = hasToc && tocNavFeatures.has('float') && (forServe || forStandalone);
  const floatButtonHtml = showFloat
    ? `<a id="back-to-toc-btn" href="#md-toc-top" aria-label="Back to Table of Contents">\u2191 TOC</a>`
    : '';
  const floatButtonScript = showFloat ? `
  <script>
    (function() {
      var toc = document.getElementById('md-toc-top');
      var btn = document.getElementById('back-to-toc-btn');
      if (!toc || !btn) return;
      new IntersectionObserver(function(entries) {
        btn.classList.toggle('visible', !entries[0].isIntersecting);
      }, { threshold: 0 }).observe(toc);
    })();
  </script>` : '';

  // Hash scroll: after dynamic content loads, scroll to the fragment target.
  // Browsers process hash fragments before JS runs, so dynamically-rendered
  // content (gisthost.github.io, SPAs) misses the initial scroll.
  const hashScrollScript = (forServe || forStandalone) ? `
  <script>
    (function() {
      if (!location.hash) return;
      var target = document.querySelector(location.hash);
      if (target) target.scrollIntoView();
    })();
  </script>` : '';

  // Copy handler: restore ASCII angle brackets and strip sr-only text on clipboard
  // Document-level copy handler: restore ASCII angle brackets and strip sr-only text.
  // Fires on any copy, not just code blocks, since inline <code> is also processed.
  const spokenContentCopyScript = (opts['macos-spoken-content'] && (forServe || forStandalone)) ? `
  <script>
    document.addEventListener('copy', function(e) {
      var sel = window.getSelection().toString();
      if (sel.indexOf('\uFF1C') === -1 && sel.indexOf('\uFF1E') === -1) return; // no fullwidth chars, don't interfere
      sel = sel.replace(/\uFF1C/g, '<').replace(/\uFF1E/g, '>');
      // Strip sr-only pronunciation text only when adjacent to operators (avoids false positives on prose)
      sel = sel.replace(/> greater than /g, '> ').replace(/< less than /g, '< ').replace(/=> arrow /g, '=>');
      e.clipboardData.setData('text/plain', sel);
      e.preventDefault();
    });
  </script>` : '';

  // Mermaid client-side rendering: needed for serve mode (via HTTP) and standalone HTML (inlined)
  const hasMermaidBlocks = /language-mermaid/.test(resolvedBody);
  const mermaidInitScript = `
  <script>
    mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });
    document.querySelectorAll('pre > code.language-mermaid').forEach(code => {
      const pre = code.parentElement;
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      pre.replaceWith(div);
    });
    mermaid.run({ querySelector: '.mermaid' });
  </script>`;

  let mermaidScripts = '';
  if (hasMermaidBlocks && forServe) {
    // Serve mode: load mermaid via HTTP
    mermaidScripts = `<script src="/mermaid.min.js"></script>${mermaidInitScript}`;
  } else if (hasMermaidBlocks && forStandalone) {
    // Standalone HTML: load mermaid from CDN
    mermaidScripts = `<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>${mermaidInitScript}`;
  }

  // Output metadata: timestamp (top-right) and filepath (bottom-left)
  // Gated on --show-timestamp / --show-filepath flags (set via CLI or config)
  const showTimestamp = opts['show-timestamp'];
  const showFilepath = opts['show-filepath'];
  const displayPath = inputPath.replace(homedir(), '~')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const fileMtime = showTimestamp ? statSync(inputPath).mtime.toISOString() : '';
  const buildTimeStr = showTimestamp ? new Date().toLocaleString() : '';

  // Timestamp element + script
  let timestampHtml = '';
  let timestampScript = '';
  if (showTimestamp && forServe) {
    // Serve mode: live-updating relative time
    timestampHtml = '<div id="serve-timestamp"></div>';
    timestampScript = `
    <script>
      (function() {
        var el = document.getElementById('serve-timestamp');
        if (!el) return;
        var mtime = new Date('${fileMtime}');
        function fmt(d) {
          return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }
        function relative(ms) {
          var s = Math.floor(ms / 1000);
          if (s < 5) return 'just now';
          if (s < 60) return s + 's ago';
          var m = Math.floor(s / 60);
          if (m < 60) return m + 'm ' + (s % 60) + 's ago';
          return Math.floor(m / 60) + 'h ' + (m % 60) + 'm ago';
        }
        function update() {
          el.textContent = 'Updated ' + fmt(mtime) + ' (' + relative(Date.now() - mtime.getTime()) + ')';
        }
        update();
        setInterval(update, 5000);
      })();
    </script>`;
  } else if (showTimestamp && forStandalone) {
    // Standalone/gist: static build time
    timestampHtml = `<div id="serve-timestamp">Generated ${buildTimeStr}</div>`;
  }

  // Filepath — clickable to reveal in Finder in serve mode (via /reveal endpoint), plain text otherwise
  // Safari blocks file:// links from all contexts (http AND local file), so /reveal is the only option
  const filepathHtml = showFilepath
    ? (forServe
      ? `<div id="serve-filepath"><a href="#" onclick="fetch('/reveal').then(r=>{if(!r.ok)alert('Could not reveal file')});return false">${displayPath}</a></div>`
      : `<div id="serve-filepath">${displayPath}</div>`)
    : '';

  // PDF pageless mode: inline metadata at the bottom of the document body
  // (position:fixed doesn't work in Puppeteer PDF — it anchors to viewport, not document bottom)
  if (!forServe && !forStandalone && (showTimestamp || showFilepath)) {
    const parts = [];
    if (showTimestamp) parts.push(`Generated ${buildTimeStr}`);
    if (showFilepath) parts.push(displayPath);
    resolvedBody += `<div style="margin-top: 40px; padding-top: 12px; border-top: 1px solid #e0e0e0; color: #999; font-size: 11px; font-family: system-ui, sans-serif;">${parts.join(' \u00B7 ')}</div>`;
  }

  const sseScript = forServe ? `
  <script>
    const evtSource = new EventSource('/events');
    evtSource.addEventListener('reload', () => location.reload());
  </script>` : '';

  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  ${(forServe || forStandalone) ? '<meta name="viewport" content="width=device-width, initial-scale=1">' : ''}
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
      ${(forServe || forStandalone) ? `width: 100%; max-width: ${pageWidth}px;` : `width: ${pageWidth}px; max-width: ${pageWidth}px;`}
      margin: 0 auto;
      padding: ${(forServe || forStandalone) ? '20px' : '40px 40px 0px 40px'};
      padding-bottom: 0px !important;
      margin-bottom: 0px !important;
      box-sizing: border-box;
    }
    img {
      max-width: 100%;
      height: auto;
    }
    ${tocNavFeatures.has('smooth') ? `
    /* Smooth scrolling for TOC navigation (--toc-nav smooth) */
    html { scroll-behavior: smooth; }` : ''}
    ${(forServe || forStandalone) ? `
    /* Mobile responsive overrides for serve/standalone mode */
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
    /* Heading back-links (--toc-nav backlinks) */
    a.heading-backlink {
      color: inherit;
      text-decoration: none;
    }
    a.heading-backlink:hover::after {
      content: ' \\2191';
      font-size: 0.6em;
      vertical-align: super;
      opacity: 0.4;
    }
    /* Bidirectional TOC flash — highlight target on both TOC→heading and heading→TOC clicks */
    .md-toc li:target,
    h1:target, h2:target, h3:target, h4:target, h5:target, h6:target {
      animation: toc-flash 1.5s ease-out;
    }
    @keyframes toc-flash {
      from { background-color: rgba(65, 131, 196, 0.25); }
      to { background-color: transparent; }
    }
    /* Floating "↑ TOC" button (--toc-nav float) */
    #back-to-toc-btn {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: #4183c4;
      color: #fff;
      padding: 8px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      text-decoration: none;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.3s ease;
      z-index: 1000;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #back-to-toc-btn.visible { opacity: 1; pointer-events: auto; }
    #back-to-toc-btn:hover { background: #3572a5; text-decoration: none; color: #fff; }
    @media print { #back-to-toc-btn { display: none !important; } }
    /* Output metadata: timestamp (top-right) and filepath (bottom-left) */
    #serve-timestamp, #serve-filepath a {
      font-size: 11px;
      color: #999;
      font-family: system-ui, -apple-system, sans-serif;
      opacity: 0.7;
      transition: opacity 0.3s ease, color 0.3s ease;
      text-decoration: none;
    }
    #serve-timestamp:hover, #serve-filepath a:hover {
      opacity: 1;
      color: #555;
    }
    #serve-timestamp {
      position: fixed;
      top: 8px;
      right: 12px;
      z-index: 999;
    }
    #serve-filepath {
      position: fixed;
      bottom: 8px;
      left: 12px;
      z-index: 999;
    }
    /* macOS Spoken Content: fullwidth angle bracket sizing (--macos-spoken-content) */
    .tts-bracket {
      display: inline-block;
      transform: scaleX(0.85);
      margin-left: -0.15em;
      margin-right: -0.15em;
    }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
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
${floatButtonHtml}
${(forServe || forStandalone) ? timestampHtml : ''}
${(forServe || forStandalone) ? filepathHtml : ''}
</body>
${mermaidScripts}
${sseScript}
${timestampScript}
${floatButtonScript}
${spokenContentCopyScript}
${hashScrollScript}
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
      // Pass 1: Measure the total content height.
      // Pass 2: Generate the PDF at that exact height.

      // Measure the body's full rendered height. Prior versions iterated
      // document.body.children and took Math.max of each child's
      // getBoundingClientRect().bottom — that misses two contributions
      // Chromium DOES honor when laying out the PDF page:
      //   1. body's own padding-bottom
      //   2. the last child's margin-bottom (not absorbed by margin collapse
      //      at the body boundary with non-zero body padding)
      // Together these are ~31pt on our themes; underreporting that much made
      // Chromium paginate content that measured as fitting. body's bottom rect
      // includes both.
      const contentBottom = await page.evaluate(() => {
        // Force a final layout pass in case any late reflow is pending.
        void document.body.offsetHeight;
        return Math.ceil(document.body.getBoundingClientRect().bottom);
      });

      // Tiny buffer to absorb sub-pixel rounding between CSS px and Chromium's
      // PDF layout. 0 works in practice, but 10 CSS px (~7.5 pt in the PDF)
      // provides a small safety net without visible whitespace.
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

async function startServer(initialHtml = null) {
  const host = opts.host;
  let currentHtml = initialHtml || await buildHtml({ forServe: true });
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

    // GET /reveal — open the source file in Finder (localhost only, macOS)
    if (pathname === '/reveal') {
      const remoteAddr = req.socket.remoteAddress;
      const isLocalhost = remoteAddr === '127.0.0.1' || remoteAddr === '::1' || remoteAddr === '::ffff:127.0.0.1';
      if (!isLocalhost) {
        res.writeHead(403, { 'Content-Type': 'text/plain' });
        res.end('Forbidden: reveal only available from localhost');
        return;
      }
      try {
        execFileSync('open', ['-R', inputPath], { stdio: 'ignore' });
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('OK');
      } catch {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Failed to reveal file');
      }
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
      // Bonjour/mDNS hostname — any Apple device (or Linux with Avahi) on the same WiFi can resolve this
      const hostname = require('os').hostname();
      const bonjourName = hostname.endsWith('.local') ? hostname : hostname + '.local';
      console.log(`  Bonjour: http://${bonjourName}:${port}`);
    } else {
      console.log('  Network: use --host 0.0.0.0 to expose');
    }
    // Auto-launch browser if --launch was passed (undefined = not passed, any string = passed)
    if (opts.launch !== undefined) {
      const url = `http://localhost:${port}`;
      const appMap = {
        safari: 'Safari', chrome: 'Google Chrome',
        chromium: 'Chromium', firefox: 'Firefox',
      };
      const app = appMap[opts.launch.toLowerCase()];
      const args = app ? ['-a', app, url] : [url];
      try { execFileSync('open', args, { stdio: 'ignore' }); }
      catch { console.error(`Error: could not launch browser '${opts.launch}'.`); }
    }

    console.log('Watching for changes... (Ctrl+C to stop)');
  });

  // File watcher with debounce.
  // Watch parent DIRECTORIES (not file paths) to survive atomic-rename saves.
  // Editors like VS Code and `touch` on some systems replace file inodes, which
  // kills path-based watchers silently. Directory watchers observe entries and
  // survive rename operations.
  let debounceTimer = null;
  const watchedDirs = new Map();  // dir → Set of filenames to watch
  const addWatch = (path) => {
    const d = dirname(path);
    if (!watchedDirs.has(d)) watchedDirs.set(d, new Set());
    watchedDirs.get(d).add(basename(path));
  };
  addWatch(inputPath);
  if (existsSync(stylesheetPath)) addWatch(stylesheetPath);

  const rebuild = () => {
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
  };

  const watchers = [];
  for (const [dir, names] of watchedDirs) {
    watchers.push(fsWatch(dir, (eventType, changedFile) => {
      if (changedFile && names.has(changedFile)) rebuild();
    }));
  }

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
  // Fail-fast: detect GitHub images that need auth in serve mode
  // Reuse this initial build as startServer()'s first render to avoid building twice
  let initialHtml = null;
  if (!opts['embed-images'] && !opts['accept-broken-images']) {
    initialHtml = await buildHtml({ forServe: true });
    const githubImages = [];
    const ghImgRe = /<img\s[^>]*src="(https:\/\/github\.com\/user-attachments\/[^"]+)"/gi;
    let ghm;
    while ((ghm = ghImgRe.exec(initialHtml)) !== null) githubImages.push(ghm[1]);
    if (githubImages.length > 0) {
      console.error('Error: Document references GitHub images that require authentication to view in the browser.');
      console.error(`  Found: ${githubImages.slice(0, 3).join(', ')}${githubImages.length > 3 ? ` (+${githubImages.length - 3} more)` : ''}`);
      console.error('');
      console.error('Resolve with one of:');
      console.error('  --embed-images            Embed remote images as base64 via gh auth');
      console.error('  --accept-broken-images    Proceed anyway (images may not load)');
      process.exit(1);
    }
  }

  // Serve mode: live preview with auto-reload
  console.log(`Input:      ${inputPath}`);
  console.log(`Stylesheet: ${stylesheetPath}`);
  console.log(`Width:      ${pageWidth}px`);
  startServer(initialHtml);
} else if (opts.html) {
  // HTML output mode: write self-contained responsive HTML file
  console.log(`Input:      ${inputPath}`);
  console.log(`Stylesheet: ${stylesheetPath}`);
  console.log(`Width:      ${pageWidth}px`);
  let htmlContent = await buildHtml({ forStandalone: true });

  // Fail-fast: detect local images that will break in the output
  const hasImageResolution = opts['embed-images'] || opts['copy-assets'] || opts['absolute-paths'] || opts['accept-broken-images'];
  if (!hasImageResolution) {
    const localImages = detectLocalImages(htmlContent);
    if (localImages.length > 0) {
      const outputDirDiffers = dirname(resolve(outputPath)) !== dirname(resolve(inputPath));
      if (opts['secret-gist']) {
        console.error('Error: Local images detected but gist HTML cannot reference local files.');
        console.error(`  Found: ${localImages.join(', ')}`);
        console.error('');
        console.error('Resolve with one of:');
        console.error('  --embed-images            Embed images as base64 data URIs (self-contained)');
        console.error('  --accept-broken-images    Proceed anyway (images will not resolve)');
        process.exit(1);
      } else if (outputDirDiffers) {
        console.error('Error: Output directory differs from source. Local images will not resolve.');
        console.error(`  Source: ${dirname(resolve(inputPath))}/`);
        console.error(`  Output: ${dirname(resolve(outputPath))}/`);
        console.error(`  Found:  ${localImages.join(', ')}`);
        console.error('');
        console.error('Resolve with one of:');
        console.error('  --embed-images            Embed images as base64 data URIs (single portable file)');
        console.error('  --copy-assets             Copy images alongside output HTML');
        console.error('  --absolute-paths          Rewrite image paths to absolute (works locally, not portable)');
        console.error('  --accept-broken-images    Proceed anyway (images will not resolve)');
        process.exit(1);
      }
    }
  }

  // Apply image resolution strategy (if not already embedded inside buildHtml)
  if (opts['copy-assets']) {
    const result = copyLocalAssets(htmlContent, inputDir, outputPath);
    htmlContent = result.html;
    if (result.count > 0) {
      console.log(`Copied:     ${result.count} image${result.count !== 1 ? 's' : ''} to ${basename(result.assetsDir)}/`);
    }
  } else if (opts['absolute-paths']) {
    htmlContent = rewriteToAbsolutePaths(htmlContent, inputDir);
  }

  // Upload to secret gist if requested
  if (opts['secret-gist']) {
    const gistFilename = basename(inputPath, extname(inputPath)) + '.html';
    const tmpFile = join(tmpdir(), gistFilename);
    writeFileSync(tmpFile, htmlContent);

    try {
      let gistUrl, gistId, username;

      if (opts['gist-id']) {
        // Update existing gist
        gistId = opts['gist-id'].includes('/')
          ? opts['gist-id'].split('/').filter(Boolean).pop()
          : opts['gist-id'];
        execSync(`gh gist edit ${gistId} -a "${tmpFile}"`, {
          stdio: ['pipe', 'pipe', 'pipe'], timeout: 30000,
        });
        username = execSync(`gh api gists/${gistId} --jq '.owner.login'`, {
          stdio: ['pipe', 'pipe', 'pipe'],
        }).toString().trim();
        gistUrl = `https://gist.github.com/${username}/${gistId}`;
      } else {
        // Create new secret gist
        const result = execSync(
          `gh gist create --desc "markdown-kit: ${gistFilename}" "${tmpFile}"`,
          { stdio: ['pipe', 'pipe', 'pipe'], timeout: 30000 },
        ).toString().trim();
        gistUrl = result;
        const parts = gistUrl.replace(/\/$/, '').split('/');
        gistId = parts.pop();
        username = parts.pop();
      }

      // Viewer: gisthost.github.io — client-side gist renderer, no CORS proxy
      // Scripts execute natively via document.write(). No data sent to non-GitHub servers.
      // Source: https://github.com/gisthost/gisthost.github.io (MIT, Simon Willison + Leon Huang)
      // Alt: htmlpreview.github.io routes content through third-party CORS proxy (api.codetabs.com)
      const viewUrl = `https://gisthost.github.io/?${gistId}/${gistFilename}`;

      // Copy to clipboard (platform-aware, non-fatal)
      try {
        if (process.platform === 'darwin') {
          execSync('pbcopy', { input: viewUrl, stdio: ['pipe', 'pipe', 'pipe'] });
        } else if (process.platform === 'win32') {
          execSync('clip', { input: viewUrl, stdio: ['pipe', 'pipe', 'pipe'] });
        }
      } catch { /* clipboard unavailable */ }

      // Output
      const action = opts['gist-id'] ? 'Updated' : 'Shared (secret gist)';
      console.log(`${action}:`);
      console.log(`  View:  ${viewUrl}`);
      console.log(`  Gist:  ${gistUrl}`);
      if (process.platform === 'darwin' || process.platform === 'win32') {
        console.log('URL copied to clipboard.');
      }

      // Open in browser if --launch
      if (opts.launch !== undefined) {
        const appMap = { safari: 'Safari', chrome: 'Google Chrome', chromium: 'Chromium', firefox: 'Firefox' };
        const app = appMap[opts.launch.toLowerCase()];
        const args = app ? ['-a', app, viewUrl] : [viewUrl];
        try { execFileSync('open', args, { stdio: 'ignore' }); } catch {}
      }
    } finally {
      unlinkSync(tmpFile);
    }
  }

  // Write local file only if -o was explicitly specified (or no --secret-gist)
  if (opts.output || !opts['secret-gist']) {
    writeFileSync(outputPath, htmlContent);
    const sizeKB = (readFileSync(outputPath).length / 1024).toFixed(1);
    console.log(`Output:     ${outputPath}`);
    console.log(`Format:     HTML (self-contained, responsive)`);
    console.log(`Size:       ${sizeKB} KB`);
  }
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
