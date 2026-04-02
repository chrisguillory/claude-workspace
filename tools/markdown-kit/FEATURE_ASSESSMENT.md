# markdown-kit Feature Assessment

Comprehensive research into visual, spatial, and informational rendering capabilities for markdown-kit. Goal: an AI generates rich content and markdown-kit renders it beautifully for human consumption in PDFs and live preview.

**Assessment Date**: March 2026 (last cleaned: March 31, 2026)
**Current Capabilities**: Mermaid, KaTeX (incl. mhchem), syntax-highlighted code, GFM tables, GFM alerts, footnotes, TOC, emoji, highlight, sub/sup, images, inline SVG, Graphviz/DOT, Vega-Lite charts, custom containers (:::), variable substitution ({{key}}), mobile responsive serve, --mobile flag, -o/--output custom path, page numbers/headers/footers (paged mode), PDF bookmarks/outline, PDF metadata from front matter, clickable internal links (TOC, footnotes), front matter handling (strip/render/raw)

---

## Backlog

| # | Feature | Category | Priority | Effort | PDF | Serve | Status |
|---|---------|----------|----------|--------|-----|-------|--------|
| 1 | QR code generation | Media | **High** | Small | Yes | Yes | Not started |
| 2 | D2 diagrams | Diagrams | **High** | Medium | Yes | Yes | Not started |
| 3 | Markmap mind maps | Info Display | **High** | Medium | Yes | Yes | Not started |
| 4 | Extended tables (colspan) | Tables | **High** | Small | Yes | Yes | Not started |
| 5 | Image captions | Media | **High** | Small | Yes | Yes | Not started |
| 6 | CSV auto-tables | GitHub | **High** | Small | Yes | Yes | Not started |
| 7 | Railroad diagrams | Diagrams | **High** | Small | Yes | Yes | Not started |
| 8 | ECharts charts | Data Viz | **High** | Medium | Yes | Yes | Not started |
| 9 | Nomnoml UML | Diagrams | **High** | Small | Yes | Yes | Not started |
| 10 | WaveDrom timing | Diagrams | **High** | Small | Yes | Yes | Not started |
| 11 | Sparklines | Data Viz | **High** | Small | Yes | Yes | Not started |
| 12 | File inclusion | Document | **High** | Small | Yes | Yes | Not started |
| 13 | Cross-references | Document | **High** | Medium | Yes | Yes | Not started |
| 14 | Barcode generation | Media | Medium | Small | Yes | Yes | Not started |
| 15 | PlantUML | Diagrams | Medium | Large | Yes | Yes | Not started |
| 16 | Excalidraw | Diagrams | Medium | Large | Yes | Yes | Not started |
| 17 | Kroki (universal) | Diagrams | Medium | Medium | Yes | Yes | Not started |
| 18 | Chart.js charts | Data Viz | Medium | Medium | Yes | Yes | Not started |
| 19 | Observable Plot | Data Viz | Medium | Medium | Yes | Yes | Not started |
| 20 | Bytefield diagrams | Diagrams | Medium | Small | Yes | Yes | Not started |
| 21 | Svgbob ASCII art | Diagrams | Medium | Small | Yes | Yes | Not started |
| 22 | SMILES molecules | Science | Medium | Medium | Yes | Yes | Not started |
| 23 | Musical notation (ABC) | Science | Medium | Medium | Yes | Yes | Not started |
| 24 | Table captions | Tables | Medium | Trivial | Yes | Yes | Not started |
| 25 | Progress bars | Info Display | Medium | Trivial | Yes | Yes | Not started |
| 26 | Badges/shields | Info Display | Medium | Small | Yes | Yes | Not started |
| 27 | Tabs (serve mode) | Document | Medium | Medium | No | Yes | Not started |
| 28 | Collapsible sections | Document | Medium | Small | Partial | Yes | Not started |
| 29 | Icon sets (FA/Material) | Media | Medium | Medium | Yes | Yes | Not started |
| 30 | Anchor links on headings | GitHub | Medium | Trivial | No | Yes | Not started |
| 31 | GeoJSON maps | Geographic | Medium | Large | Yes | Yes | Not started |
| 32 | Regex visualization | CS Viz | Medium | Medium | Yes | Yes | Not started |
| 33 | Cytoscape.js graphs | CS Viz | Medium | Large | Yes | Yes | Not started |
| 34 | VexFlow notation | Science | Low | Medium | Yes | Yes | Not started |
| 35 | Ditaa ASCII art | Diagrams | Low | Medium | Yes | Yes | Not started |
| 36 | Circuit diagrams | Science | Low | Large | Yes | Yes | Not started |
| 37 | Plotly charts | Data Viz | Low | Large | Yes | Yes | Not started |
| 38 | Timeline visualization | Info Display | Low | N/A | Yes | Yes | Mermaid has this |
| 39 | Sankey diagrams | Info Display | Low | N/A | Yes | Yes | Mermaid has this |
| 40 | Treemaps | Info Display | Low | Large | Yes | Yes | Not started |
| 41 | Heatmaps | Info Display | Low | Large | Yes | Yes | Not started |
| 42 | Bibliography/citations | Document | Low | Large | Yes | Yes | Not started |
| 43 | Glossary | Document | Low | Medium | Yes | Yes | Not started |
| 44 | Image galleries | Media | Low | Medium | Yes | Yes | Not started |
| 45 | Before/after sliders | Media | Low | Small | No | Yes | Not started |
| 46 | 3D STL rendering | 3D | Niche | Large | No | Yes | Not started |
| 47 | Three.js static | 3D | Niche | Large | No | Yes | Not started |
| 48 | 3Dmol.js molecules | 3D | Niche | Large | No | Yes | Not started |
| 49 | Feynman diagrams | Science | Niche | Large | Partial | Partial | Not started |
| 50 | Astronomical charts | Science | Niche | Large | Maybe | Maybe | Not started |
| 51 | Periodic table | Science | Niche | Medium | Yes | Yes | Not started |
| 52 | Kanban boards | Info Display | Niche | N/A | Yes | Yes | Mermaid has this |
| 53 | Org charts | Info Display | Niche | N/A | Yes | Yes | Mermaid has this |

---

## 1. Diagram Types Beyond Mermaid

### 1.1 D2 (Modern Declarative Diagrams)

**What**: A modern diagram scripting language (d2lang.com) designed specifically for software architecture diagrams. Supports connections, containers, classes, sequence diagrams, and has multiple layout engines (Dagre, ELK, TALA).

**Popularity**: ~12k GitHub stars. Growing fast among infrastructure and DevOps teams. Created by Terrastruct.

**JS Library**: `@terrastruct/d2` -- official npm package. Built on WASM (Go compiled to WASM). Works in browser and Node.js. Renders to SVG. Uses web workers for the WASM execution.

**Implementation**: Add `@terrastruct/d2` as a dependency. Register a `marked` extension for ` ```d2 ` code blocks. In the Puppeteer page, load the D2 WASM module and render source to SVG. The library is ESM-compatible and handles layout internally.

**Effort**: Medium. The WASM library is larger than Graphviz and may need web worker setup. But the API is clean.

**Value**: High. D2's syntax is more intuitive than DOT for software architecture. AI models already know D2 syntax well. It produces visually cleaner output than Mermaid for architecture diagrams.

**Priority**: **High**

---

### 1.2 PlantUML

**What**: Comprehensive UML diagramming tool. Supports class, sequence, activity, component, state, object, deployment, timing, and many more diagram types. Industry standard for software documentation.

**Popularity**: Extremely popular. >10k GitHub stars. The most complete UML tool available.

**JS Library**: `plantuml.js` -- official pure-JS port using CheerpJ (Java-to-WASM compiler). Available on GitHub. However, it is very large (the full PlantUML Java codebase compiled to WASM/JS) and requires an HTTP server context (cannot run from `file://`). Alternative: use Kroki API as a proxy.

**Implementation**: Two approaches:
1. **Via Kroki** (recommended): Send PlantUML source to kroki.io API or self-hosted Kroki, get SVG back. Medium effort, small binary size.
2. **Native plantuml.js**: Include the WASM bundle. Very large download (~50MB+). Works offline.

**Effort**: Large (native) or Medium (via Kroki).

**Value**: High for teams that already use PlantUML. However, Mermaid covers most of the same diagram types with simpler syntax.

**Priority**: **Medium**. Mermaid already covers 80% of the use cases. PlantUML matters for teams with existing PlantUML assets.

---

### 1.3 Excalidraw (Hand-Drawn Style)

**What**: Hand-drawn style diagrams with a distinctive informal aesthetic. JSON-based scene format. Popular for whiteboard-style technical diagrams.

**Popularity**: ~95k GitHub stars. Enormously popular.

**JS Library**: `@excalidraw/excalidraw` -- the full editor. `excalidraw-to-svg` -- Node.js library for rendering JSON to SVG. However, server-side rendering requires headless Chromium (the library uses React and DOM APIs internally).

**Implementation**: Since we already have Puppeteer/Chromium in our pipeline, we could load the Excalidraw renderer in the page context. Register ` ```excalidraw ` code blocks, parse the JSON scene data, and render to SVG using the Excalidraw library.

**Effort**: Large. Excalidraw is a React component and does not support true server-side rendering. Must be rendered in the browser context.

**Value**: Medium-High. The hand-drawn aesthetic is distinctive and popular, but AI generating Excalidraw JSON is more complex than generating Mermaid or DOT text.

**Priority**: **Medium**

---

### 1.4 Kroki (Universal Diagram Server)

**What**: A unified HTTP API that supports 20+ diagram formats. Acts as a proxy to render diagrams from text. Supports: BlockDiag, BPMN, Bytefield, C4 (PlantUML), Ditaa, Erd, Excalidraw, Graphviz, Mermaid, Nomnoml, PlantUML, Structurizr, SvgBob, Symbolator, UMLet, Vega, Vega-Lite, WaveDrom, WireViz, and more.

**Popularity**: ~3.9k GitHub stars. MIT license. Active development.

**JS Library**: No JS library needed -- it is an HTTP API. Diagrams are sent as POST requests (plain text body) or GET requests (deflate+base64 encoded URL). Returns SVG/PNG/JPEG.

**Implementation**: Register code blocks for any supported format (` ```plantuml `, ` ```ditaa `, etc.). During rendering, POST the source to `https://kroki.io/{format}/svg` and inject the returned SVG. Can self-host via Docker for offline use.

**Effort**: Medium. Need HTTP client in the rendering pipeline. Must handle network availability (online vs offline). Could make the Kroki endpoint configurable.

**Value**: Very high as a fallback/universal solution. Instead of embedding 20 different WASM libraries, use one API for all niche formats.

**Priority**: **Medium**. Best as a fallback for formats we do not support natively.

---

### 1.5 Nomnoml

**What**: Simple, sassy UML class diagram renderer. Clean syntax focused on boxes and arrows with styling.

**Popularity**: ~8.8k GitHub stars. Stable and well-maintained.

**JS Library**: `nomnoml` -- npm package. Pure JavaScript. Only dependency is `graphre` (graph layout engine). Renders to SVG. Has CLI via `npx nomnoml`. Works in Node.js.

**Implementation**: Add `nomnoml` as a dependency. Register ` ```nomnoml ` code blocks. Call `nomnoml.renderSvg(source)` and inject the result. Very straightforward.

**Effort**: Small. Clean API, small library, no WASM or browser DOM needed.

**Value**: Medium. Nice for quick class diagrams, but Mermaid class diagrams cover similar territory.

**Priority**: **High** (because the effort is so small for the value gained)

---

### 1.6 WaveDrom (Digital Timing Diagrams)

**What**: Digital timing diagram (waveform) rendering. Uses WaveJSON format to describe signal waveforms. Standard tool for hardware/IC engineering.

**Popularity**: ~2.7k GitHub stars. Industry standard for hardware documentation.

**JS Library**: `wavedrom` -- npm package. Renders to SVG using HTML5. Works in browser and Node.js. CLI tool available.

**Implementation**: Add `wavedrom` as a dependency. Register ` ```wavedrom ` code blocks. Parse the WaveJSON and render to SVG. Kroki also supports WaveDrom as a backend.

**Effort**: Small.

**Value**: Niche but irreplaceable for hardware engineering docs. No other tool does timing diagrams well.

**Priority**: **High** (small effort, unique capability)

---

### 1.7 Bytefield (Protocol/Memory Layout Diagrams)

**What**: Generates byte field diagrams showing packet structures, memory layouts, protocol headers. Inspired by the LaTeX bytefield package.

**Popularity**: ~200 GitHub stars. Niche but well-maintained by Deep Symmetry.

**JS Library**: `bytefield-svg` -- npm package. Node.js CLI and API. Uses a Clojure DSL (via SCI, the Small Clojure Interpreter compiled to JS). Outputs SVG.

**Effort**: Small. `npm install bytefield-svg`, call the API with source text, get SVG back.

**Value**: Niche. Useful for networking, systems programming, and protocol documentation.

**Priority**: **Medium**

---

### 1.8 Railroad Diagrams (Grammar/Syntax Visualization)

**What**: Visualize grammars, regex patterns, and syntax rules as railroad (syntax) diagrams. The style used on json.org and in language specs.

**Popularity**: `railroad-diagrams` by Tab Atkins has ~1.8k GitHub stars. Well-established.

**JS Library**: `railroad-diagrams` -- npm package. Pure JavaScript, outputs SVG. Also `rrdiagram-js` which generates diagrams from BNF notation. No browser DOM required.

**Implementation**: Register ` ```railroad ` or ` ```bnf ` code blocks. Parse BNF/EBNF input and generate SVG. The library is small and focused.

**Effort**: Small. The API is simple and the library is lightweight.

**Value**: High for language/CS documentation. AI agents explaining grammars, protocols, or regex patterns benefit greatly from visual railroad diagrams.

**Priority**: **High**

---

### 1.9 Ditaa (ASCII Art to Clean Diagrams)

**What**: Converts ASCII art diagrams into clean, proper-looking diagrams with straight lines, corners, and filled shapes. Originally a Java tool.

**Popularity**: Classic tool, widely known. No pure JS port -- requires Kroki or Java runtime.

**JS Library**: No pure JS implementation. Available via Kroki API.

**Implementation**: Via Kroki only. Register ` ```ditaa ` code blocks and POST to Kroki.

**Effort**: Medium (requires Kroki integration).

**Value**: Low-Medium. ASCII art diagrams are charming but Mermaid/DOT are more practical for AI generation.

**Priority**: **Low**

---

### 1.10 Svgbob (ASCII Art to SVG)

**What**: Converts ASCII art to clean SVG. More capable than Ditaa, understands curves and complex shapes. Written in Rust.

**Popularity**: ~3.8k GitHub stars for the Rust version.

**JS Library**: `svgbob-wasm` -- npm package. WASM compilation of the Rust library. However, only 22 weekly downloads and last published 4 years ago. Low maintenance.

**Implementation**: Add `svgbob-wasm` dependency. Register ` ```svgbob ` code blocks. Or use via Kroki API.

**Effort**: Small (WASM) or Medium (Kroki).

**Value**: Medium. Nice for converting hand-typed ASCII art, but the WASM package is poorly maintained.

**Priority**: **Medium** (via Kroki), **Low** (native WASM due to maintenance concerns)

---

## 2. Data Visualization / Charts

### 2.1 Chart.js

**What**: Simple, flexible JavaScript charting. Supports bar, line, pie, doughnut, radar, scatter, bubble, polar area charts.

**Popularity**: ~65.2k GitHub stars. Most popular JS charting library by stars.

**JS Library**: `chart.js` -- npm package. Server-side rendering via `chartjs-node-canvas` (uses node-canvas, a native dependency). Outputs PNG/JPEG.

**Implementation**: Requires native `canvas` dependency (C++ compilation). Register ` ```chartjs ` code blocks with JSON config. Render via `chartjs-node-canvas` to get a buffer, then embed as a base64 image.

**Effort**: Medium. The native `canvas` dependency is the main complication. Chart.js outputs raster images (PNG), not SVG.

**Value**: Medium. Chart.js is popular but its config format is verbose. Vega-Lite is more suitable for AI generation due to its declarative grammar.

**Priority**: **Medium**

---

### 2.2 Apache ECharts

**What**: Comprehensive charting library from Apache. Supports line, bar, scatter, pie, candlestick, map, funnel, gauge, treemap, sunburst, parallel, sankey, and dozens more chart types. Used extensively in China/Asia.

**Popularity**: ~62.5k GitHub stars. Second-most popular JS charting library.

**JS Library**: `echarts` -- npm package. **Native SSR SVG support** since v5.3.0. In Node.js: `echarts.init(null, null, { renderer: 'svg', ssr: true, width, height })` then `chart.renderToSVGString()`. No DOM, no canvas, no native dependencies. Lightweight client runtime is only 4KB gzipped.

**Implementation**: Add `echarts` as a dependency. Register ` ```echarts ` code blocks with JSON options. Call `echarts.init()` in SSR mode, set options from JSON, call `renderToSVGString()`, inject SVG. Clean and straightforward.

**Effort**: Medium. Library is large (~1MB) but the SSR API is clean. No native dependencies.

**Value**: High. Enormous chart type variety. JSON-based options format is well-suited for AI generation.

**Priority**: **High**

---

### 2.3 Observable Plot

**What**: Modern, concise charting library from the D3 team. Emphasizes "exploratory data analysis" with a compact API. Supports bar, line, dot, area, rule, tick, cell, rect, text, arrow, geo, and more.

**Popularity**: ~4k GitHub stars. Growing among data scientists.

**JS Library**: `@observablehq/plot` -- npm package. **Server-side SVG rendering works** via JSDOM. Pass a JSDOM `document` instance via the `document` option, then serialize the output via `outerHTML`.

**Implementation**: Add `@observablehq/plot` and `jsdom` as dependencies. Register ` ```plot ` code blocks. Parse the JS specification, create the plot with JSDOM, serialize to SVG string.

**Effort**: Medium. Requires JSDOM setup. The specification format is JavaScript (not pure JSON), which makes it harder for AI to generate safely.

**Value**: Medium. Elegant API but the JavaScript spec format (not JSON) is a disadvantage for AI generation compared to Vega-Lite.

**Priority**: **Medium**

---

### 2.4 Plotly

**What**: Scientific charting library. Supports 40+ chart types including 3D plots, statistical charts, financial charts, maps, and scientific visualizations.

**Popularity**: ~17.6k GitHub stars. Standard in scientific computing.

**JS Library**: `plotly.js` -- npm package. Server-side rendering requires either jsdom or Puppeteer. Alternative: Plotly Node Export Server, Orca CLI, or Kaleido tool. None are lightweight.

**Implementation**: Could render ` ```plotly ` code blocks in the Puppeteer page context (since we already have Chromium). Load plotly.js in the page, call `Plotly.newPlot()`, then export to SVG.

**Effort**: Large. The library is very large (~3.5MB minified). Server-side export options are complex.

**Value**: Medium. Powerful but heavyweight. Vega-Lite covers most common chart types more efficiently.

**Priority**: **Low**

---

### 2.5 Sparklines (Inline Mini-Charts)

**What**: Tiny inline charts that show trends at a glance. Commonly embedded in tables, text, or dashboards. Named by Edward Tufte.

**Popularity**: Multiple libraries. `@fnando/sparkline` (~800 stars), `sparkline-svg` on npm.

**JS Library**: `sparkline-svg` -- npm package. Zero dependencies. Generates SVG as data URIs. Works in Node.js. Also `@fnando/sparkline` for browser-side rendering.

**Implementation**: Two approaches:
1. **Inline syntax**: `{sparkline: 1,3,7,4,2,8}` -- a custom marked inline extension that generates an inline SVG sparkline.
2. **Code block**: ` ```sparkline ` with data values on separate lines.

Generate SVG string from the data array and inject inline.

**Effort**: Small. The libraries are tiny and the SVG output is compact.

**Value**: High. Sparklines are a powerful communication tool -- they pack trend information into minimal space. Perfect for tables showing data trends.

**Priority**: **High**

---

## 3. Science and Math Beyond KaTeX

### 3.1 SMILES Molecular Structure Drawing

**What**: Renders 2D chemical molecular structures from SMILES notation (Simplified Molecular Input Line Entry System). Shows bonds, atoms, rings.

**Popularity**: `smiles-drawer` -- published in Journal of Chemical Information and Modeling. Used in chemistry/pharma.

**JS Library**: `smiles-drawer` -- npm package (v2.1.7). Pure JavaScript, no dependencies. Renders to SVG or Canvas. Works client-side.

**Implementation**: Register ` ```smiles ` code blocks. In the Puppeteer page, load smiles-drawer, parse the SMILES string, render to SVG canvas, extract SVG. Could also potentially run in Node.js directly.

**Effort**: Medium.

**Value**: Medium. Valuable for chemistry and pharma documentation, but niche.

**Priority**: **Medium**

---

### 3.2 Musical Notation (ABC)

**What**: Renders sheet music from ABC notation -- a text-based music notation standard. Shows staves, notes, rests, key signatures, time signatures, chords.

**Popularity**: `abcjs` -- ~1.9k GitHub stars. The standard JS library for ABC notation.

**JS Library**: `abcjs` -- npm package. Renders to SVG. Works in browser. The `@folkdb/abc-render-svg` package specifically targets Node.js SVG rendering.

**Implementation**: Register ` ```abc ` code blocks. Load abcjs in the page context, call the rendering API, get SVG output.

**Effort**: Medium. Library is large (music rendering is complex). SVG output is clean.

**Value**: Medium. Music education and theory documentation benefits significantly.

**Priority**: **Medium**

---

### 3.3 Musical Notation (VexFlow)

**What**: Professional-quality music engraving. More powerful than abcjs -- supports tablature, percussion notation, custom voice layouts.

**Popularity**: ~3.9k GitHub stars. Used in professional music software.

**JS Library**: `vexflow` -- npm package (v5). TypeScript. Outputs to Canvas and SVG. Works in Node.js.

**Implementation**: Register ` ```vexflow ` code blocks. Requires a JavaScript-based specification (not simple text notation), making it less ideal for AI generation than ABC notation.

**Effort**: Medium.

**Value**: Low-Medium. More powerful than abcjs but harder for AI to generate. ABC notation is simpler and more practical.

**Priority**: **Low** (prefer abcjs for simplicity)

---

### 3.4 Circuit Diagrams

**What**: Electronic circuit schematics showing components (resistors, capacitors, transistors) and connections.

**Popularity**: `schemdraw` (Python only), CircuiTikZ (LaTeX only). No established pure-JS circuit diagram library.

**JS Library**: No mature pure-JS solution. Possible approaches:
1. Use Mermaid's generic flowchart with custom shapes
2. Use D2 with custom icons
3. Render via Kroki with CircuiTikZ backend
4. Use `circuit-diagram` (niche, low usage)

**Effort**: Large. No good off-the-shelf JS solution.

**Value**: Low. Niche audience (EE students/engineers). Better served by specialized tools.

**Priority**: **Low**

---

### 3.5 Feynman Diagrams

**What**: Particle physics interaction diagrams showing particle propagators and vertices.

**JS Library**: No established JS library. Typically rendered via TikZ-Feynman (LaTeX) or custom SVG. Could potentially use Graphviz with custom node/edge styles.

**Effort**: Large.

**Value**: Very niche. Physics research papers only.

**Priority**: **Niche**

---

## 4. Computer Science Visualizations

### 4.1 Regex Visualization

**What**: Visualize regular expressions as railroad diagrams showing the matching structure.

**Popularity**: Regexper.com is extremely popular. `regulex` (~5.3k stars) provides a JS library.

**JS Library**: `regulex` -- npm package. Pure JavaScript regex parser and visualizer. Outputs SVG. `regexper` is also available but is more of a web app than a library.

**Implementation**: Register ` ```regex ` code blocks. Parse the regex with regulex, generate SVG railroad diagram, inject. Clean integration.

**Effort**: Medium. Need to handle the rendering pipeline (regulex produces SVG via DOM manipulation, may need jsdom or browser context).

**Value**: High for programming education and documentation. AI explaining regex patterns with visual diagrams is extremely effective.

**Priority**: **Medium**

---

### 4.2 Graph/Network Visualization (Cytoscape.js)

**What**: Interactive graph theory library for network diagrams, biological networks, social graphs. Supports multiple layout algorithms.

**Popularity**: ~10.2k GitHub stars. Standard in bioinformatics and network analysis.

**JS Library**: `cytoscape` -- npm package. Works headlessly in Node.js for graph analysis. SVG export via `cytoscape-svg` extension.

**Implementation**: Register ` ```cytoscape ` code blocks with JSON graph data. Render in headless mode or in Puppeteer page context. Export to SVG.

**Effort**: Large. The library is complex and the headless SVG export has limitations.

**Value**: Medium. Good for network topology and graph visualization, but DOT/Graphviz is simpler for most use cases.

**Priority**: **Medium**

---

## 5. 3D Content

### 5.1 Three.js Static Rendering

**What**: 3D scene rendering to static images. Could show 3D models, geometric visualizations, or data in 3D space.

**JS Library**: `three` -- npm package (>100k stars). Server-side rendering requires `headless-gl` (native WebGL implementation for Node.js) or rendering via Puppeteer.

**Implementation**: In serve mode, load Three.js in the browser -- interactive 3D is fully possible. For PDF, render in Puppeteer and capture as a raster image (PNG), since PDFs cannot contain interactive 3D.

**Effort**: Large. headless-gl has native dependencies. Puppeteer approach is more practical.

**Value**: Low for PDF (just a flat image). Medium for serve mode (interactive 3D).

**Priority**: **Niche**

---

### 5.2 STL File Rendering

**What**: Render 3D STL files (common in 3D printing, CAD). GitHub renders these inline.

**JS Library**: Three.js + STLLoader.

**Implementation**: Serve mode only (interactive). For PDF, render a static angle as PNG via Puppeteer with Three.js loaded. Requires the STL file to be referenced, not inline.

**Effort**: Large.

**Priority**: **Niche**

---

### 5.3 3D Molecular Viewers

**What**: Interactive 3D molecule visualization. Protein structures, drug molecules.

**JS Library**: `3Dmol.js` -- npm package. WebGL-based. Browser only.

**Implementation**: Serve mode only. Register ` ```3dmol ` code blocks with PDB/SDF data or molecule identifiers.

**Effort**: Large.

**Priority**: **Niche**

---

### 5.4 Summary: 3D in PDF vs Serve Mode

3D content cannot be interactive in PDF. Options:
- **PDF**: Render a static viewpoint as a raster image (PNG/JPEG). Loss of interactivity.
- **Serve mode**: Full interactive 3D is possible via WebGL/Three.js.

Verdict: 3D features are serve-mode-only for meaningful value. Low priority overall unless the user has specific 3D documentation needs.

---

## 6. Geographic/Spatial

### 6.1 GeoJSON/Map Rendering (D3-geo)

**What**: Render geographic data (country maps, choropleth maps, point maps) from GeoJSON data using D3's geographic projection system.

**Popularity**: D3-geo is part of D3.js (~110k stars). Standard for web cartography.

**JS Library**: `d3-geo` -- npm package. Can render SVG maps server-side with JSDOM. No browser DOM required for SVG path generation (D3-geo's path generator works with any SVG-like API).

**Implementation**: Register ` ```geojson ` code blocks. Parse the GeoJSON, apply a projection (Mercator, Albers, etc.), generate SVG paths. Requires a projection/viewport specification alongside the GeoJSON data.

**Effort**: Large. Map rendering involves projection selection, viewport calculation, styling, labels. But the D3 APIs are well-documented.

**Value**: Medium. Useful for data journalism, geographic analysis, location-based documentation.

**Priority**: **Medium**

---

## 7. Table Enhancements

### 7.1 Extended Tables (Colspan/Rowspan)

**What**: Support merged cells in markdown tables using extended pipe syntax. E.g., `||` for colspan.

**JS Library**: `marked-extended-tables` -- npm package. Extends marked.js to support colspan via extra `|` characters and rowspan via `^` markers.

**Implementation**: Add `marked-extended-tables` as a dependency. Use it as a marked extension. Minimal code.

**Effort**: Small. Drop-in marked extension.

**Value**: High. Complex data tables often need merged cells. AI generating comparison tables, feature matrices, and structured data benefits from colspan/rowspan.

**Priority**: **High**

---

### 7.2 Table Captions

**What**: Add captions/titles to tables. Standard in academic/technical writing. HTML `<caption>` element.

**Implementation**: Convention: A line immediately before or after the table starting with `Table:` or `Caption:` gets wrapped in a `<caption>` element. Custom marked extension.

**Effort**: Trivial. Simple pre/post-processing of table tokens.

**Value**: Medium. Good for formal documents.

**Priority**: **Medium**

---

## 8. Information Display Patterns

### 8.1 Markmap (Mind Maps)

**What**: Convert markdown heading/list structure into interactive mind map SVGs. Input is just regular markdown -- the hierarchy creates the mind map.

**Popularity**: ~8.9k GitHub stars. Active development. VSCode extension available.

**JS Library**: `markmap-lib` + `markmap-view` -- npm packages. `markmap-lib` transforms markdown to mind map data. `markmap-view` renders SVG. Works in Node.js.

**Implementation**: Register ` ```mindmap ` code blocks (or a special command). Parse the markdown content with `markmap-lib`, render with `markmap-view` in the Puppeteer page context (uses D3 for SVG rendering).

**Effort**: Medium. Library is well-documented but rendering requires browser context for D3.

**Value**: Very high. Mind maps are an excellent visual communication tool. AI can structure complex topics as hierarchical markdown, and markmap turns it into a visual overview.

**Priority**: **High**

---

### 8.2 Progress Bars

**What**: Visual progress indicators. E.g., `[=====>    ] 60%` or a custom syntax.

**Implementation**: Custom marked extension. Syntax: `[progress: 60%]` or `[===>      ] 60%`. Renders as a styled HTML `<div>` with CSS. Pure CSS, no JS library needed.

**Effort**: Trivial. A few lines of CSS and a simple tokenizer.

**Value**: Medium. Useful for project status, completion tracking, skill indicators.

**Priority**: **Medium**

---

### 8.3 Badges/Shields

**What**: Colored badge pills showing status, version, build status. Like shields.io badges but generated locally.

**Implementation**: Custom marked syntax: `[badge: label | value | color]`. Generate inline SVG (similar to shields.io SVG format). No external library needed -- SVG template is simple.

**Effort**: Small.

**Value**: Medium. Useful for status indicators, version badges, labels.

**Priority**: **Medium**

---

### 8.4 Treemaps / Heatmaps

**What**: Space-filling visualizations showing hierarchical data (treemaps) or density/intensity data (heatmaps).

**JS Library**: D3 has both `d3-treemap` and heatmap capabilities. ECharts supports both natively.

**Implementation**: Would be covered by ECharts or Vega-Lite integration. No separate library needed.

**Effort**: N/A (covered by chart library integrations).

**Priority**: **Low** (available through chart libraries)

---

## 9. Image/Media Features

### 9.1 QR Code Generation

**What**: Generate QR codes from text, URLs, or data directly in the document. Reader scans to access links, WiFi credentials, contact info.

**Popularity**: `qrcode-svg` has ~500 stars. `qrcode` (node-qrcode) has ~7k stars.

**JS Library**: `qrcode-svg` -- npm package. Pure JavaScript, zero dependencies. Generates SVG. Also `qrcode` package which supports SVG output mode.

**Implementation**: Custom marked inline extension or code block. Syntax options:
- Inline: `{qr: https://example.com}` generates an inline QR code SVG
- Block: ` ```qr ` code block with URL/text content

Call `new QRCode({ content: text }).svg()` and inject the SVG.

**Effort**: Small. Library is tiny and API is simple.

**Value**: High. QR codes in PDFs are immediately practical -- link to resources, provide WiFi details, embed vCard data. AI generating documentation with actionable QR links is very useful.

**Priority**: **High**

---

### 9.2 Barcode Generation

**What**: Generate 1D and 2D barcodes (Code128, EAN, UPC, Data Matrix, PDF417, etc.).

**Popularity**: `bwip-js` ~2.1k stars, supports 100+ barcode formats. `JsBarcode` ~5.3k stars.

**JS Library**: `bwip-js` -- npm package. Pure JavaScript (PostScript transpiled to JS). Outputs SVG on all platforms. Supports over 100 barcode standards.

**Implementation**: Register ` ```barcode ` code blocks with type and data. Call `bwipjs.toSVG()` and inject.

**Effort**: Small.

**Value**: Medium. Useful for inventory docs, shipping labels, product catalogs.

**Priority**: **Medium**

---

### 9.3 Image Captions

**What**: Render image alt text as visible captions below images, using `<figure>` and `<figcaption>` HTML elements.

**Implementation**: Custom marked renderer that wraps `<img>` in `<figure>` and renders the alt text as `<figcaption>`. Convention: images with alt text get captions; images with empty alt text do not.

**Effort**: Small. Override the image renderer in marked.

**Value**: High. Proper figure captions are essential for technical and academic documents. AI-generated content with captioned figures looks significantly more professional.

**Priority**: **High**

---

### 9.4 Icon Sets (Font Awesome / Material Icons)

**What**: Embed scalable vector icons inline in text. `:fa-check:`, `:mdi-alert:`, etc.

**JS Library**: `@fortawesome/fontawesome-svg-core` + icon packages. The SVG core can generate inline SVG strings from icon names without a browser DOM.

**Implementation**: Custom marked inline extension. Syntax: `:fa-check:` or `:icon-name:`. Look up the icon in the Font Awesome SVG library, generate inline `<svg>` markup. Need to bundle the icon data (FA free set is ~1.6k icons).

**Effort**: Medium. Need to bundle icon data and handle the lookup. Font Awesome's SVG core is designed for this.

**Value**: Medium. Icons add visual richness to documents. Useful for feature lists, status indicators, navigation aids.

**Priority**: **Medium**

---

### 9.5 Image Galleries

**What**: Grid layout of multiple images with optional captions.

**Implementation**: CSS Grid/Flexbox layout triggered by a custom syntax. E.g., consecutive images with no text between them get arranged in a grid. Or ` ```gallery ` block with image URLs.

**Effort**: Medium (CSS layout + custom tokenizer).

**Value**: Low-Medium. Useful for photo documentation but niche.

**Priority**: **Low**

---

### 9.6 Before/After Comparison

**What**: Side-by-side or slider comparison of two images.

**Implementation**: Serve mode only (interactive slider needs JavaScript). For PDF, render as side-by-side with a divider.

**Effort**: Small for serve mode.

**Priority**: **Low** (serve mode only)

---

## 10. Document Features

### 10.1 File Inclusion

**What**: Include content from other markdown files.

```markdown
!include(common/header.md)

## Section

!include(sections/introduction.md)
```

**Implementation**: Pre-processing pass before markdown parsing. Regex for `!include(path)` or `@include(path)`. Read the referenced file, substitute inline. Handle relative paths from the including file's directory.

**Effort**: Small. File reading + string substitution. Need to handle circular includes.

**Value**: High. Composable documents from reusable parts. AI can generate modular documentation.

**Priority**: **High**

---

### 10.2 Cross-References

**What**: Reference other sections, figures, or tables by label.

```markdown
See [Section @intro] for background.
As shown in [Figure @arch-diagram], the system...
```

**Implementation**: Custom marked extension. During rendering, build a registry of headings, figures, and tables with their auto-generated or explicit labels. Replace `@label` references with links and auto-numbered text ("Section 2.1", "Figure 3").

**Effort**: Medium. Need a two-pass renderer (first pass to collect labels, second pass to resolve references).

**Value**: High. Essential for formal technical documents. AI generating structured reports benefits from automatic cross-referencing.

**Priority**: **High**

---

### 10.3 Bibliography/Citations

**What**: Academic-style citations and bibliography. Pandoc syntax: `[@smith2020]`, `[@jones2021, p. 42]`.

**Implementation**: Custom marked extension. Parse `[@key]` references. Look up keys in a BibTeX/CSL-JSON bibliography file specified in front matter. Generate numbered/author-date citations inline and a formatted bibliography at the end.

**Effort**: Large. Citation formatting is complex (CSL styles). Would need a CSL processor library like `citeproc-js`.

**Value**: Medium. Important for academic documents but not common in general technical writing.

**Priority**: **Low**

---

### 10.4 Tabs (Serve Mode)

**What**: Tabbed content panels for showing alternative views of the same information. Common in docs sites (VitePress, Docusaurus).

```markdown
:::tabs
::tab[JavaScript]
\`\`\`js
console.log("hello");
\`\`\`
::tab[Python]
\`\`\`python
print("hello")
\`\`\`
:::
```

**Implementation**: Custom containers with JavaScript for tab switching. Serve mode only (PDFs cannot have interactive tabs). For PDF, render all tabs sequentially with labels.

**Effort**: Medium.

**Value**: Medium for serve mode. Zero for PDF (would need fallback rendering).

**Priority**: **Medium** (serve mode only)

---

### 10.5 Collapsible Sections

**What**: Expandable/collapsible content sections beyond HTML `<details>/<summary>`.

**Note**: HTML `<details>` already works in our pipeline (GFM supports it). Could add ` ```details ` or `:::details` syntax for convenience.

**Implementation**: Map custom syntax to `<details><summary>` HTML. Small marked extension.

**Effort**: Small.

**Priority**: **Medium**

---

### 10.6 Conditional Content

**What**: Show or hide content based on output format (PDF vs serve) or front matter variables.

```markdown
:::if format=pdf
Download the interactive version at...
:::

:::if format=serve
Click the diagram to zoom in.
:::
```

**Implementation**: Pre-processing pass that strips blocks based on current rendering mode.

**Effort**: Small.

**Value**: Medium. Useful when the same source generates both PDF and live preview.

**Priority**: Medium

---

## 11. GitHub-Specific Features

### 11.1 CSV Auto-Rendering as Tables

**What**: ` ```csv ` code blocks automatically rendered as formatted HTML tables instead of raw text.

**Implementation**: Custom marked extension for ` ```csv ` blocks. Parse CSV (handle quoting, commas in values, headers). Generate an HTML `<table>`. No external library needed -- CSV parsing is simple.

**Effort**: Small. CSV parsing is straightforward for well-formed data.

**Value**: High. CSV is the universal data exchange format. AI generating tabular data as CSV with automatic rendering is very practical.

**Priority**: **High**

---

### 11.2 Anchor Links on Headings

**What**: Hover-to-show `#` link next to headings (GitHub style). Clicking copies the anchor URL.

**Implementation**: CSS + small JS for hover behavior. Modify heading renderer to include an anchor `<a>` element. We already generate heading IDs via `marked-gfm-heading-id`.

**Effort**: Trivial. CSS only.

**Value**: Medium. Useful in serve mode for sharing links to specific sections.

**Priority**: **Medium** (serve mode only)

---

## Implementation Strategy

### Phase 1: Small Extensions (Next)

Drop-in libraries with established marked patterns:

1. **QR codes** -- `qrcode-svg` package.
2. **Extended tables** -- `marked-extended-tables` package.
3. **Image captions** -- Override image renderer.
4. **CSV auto-tables** -- Custom code block handler. No library needed.
5. **Nomnoml** -- `nomnoml` package. Simple renderSvg() call.
6. **WaveDrom** -- `wavedrom` package. WaveJSON to SVG.
7. **Railroad diagrams** -- `railroad-diagrams` package.
8. **File inclusion** -- Pre-processing regex pass.
9. **Sparklines** -- `sparkline-svg` package.
10. **Table captions** -- Simple post-processing of table tokens.
11. **Progress bars** -- CSS + inline extension.
12. **Anchor links** -- CSS hover effect on headings. Serve mode only.

### Phase 2: Medium Integrations

Libraries requiring more setup but high value:

13. **ECharts** -- SSR SVG mode. JSON options to chart.
14. **D2 diagrams** -- WASM module in browser context.
15. **Markmap mind maps** -- markmap-lib + markmap-view.
16. **Cross-references** -- Two-pass rendering system.
17. **Barcode generation** -- `bwip-js` SVG output.
18. **SMILES molecules** -- `smiles-drawer` in browser context.
19. **ABC music notation** -- `abcjs` in browser context.
20. **Bytefield** -- `bytefield-svg` Node.js API.
21. **Regex visualization** -- `regulex` with SVG rendering.
22. **Tabs** -- Serve mode only, with PDF fallback.

### Phase 3: Large/Niche Integrations (Future)

23. **Kroki integration** -- Universal fallback for 20+ diagram formats.
24. **PlantUML** -- Via Kroki or native WASM.
25. **Excalidraw** -- React-based, needs Puppeteer rendering.
26. **GeoJSON maps** -- D3-geo with JSDOM.
27. **Bibliography/citations** -- citeproc-js integration.
28. **Icon sets** -- Font Awesome SVG core.
29. **Plotly** -- Via Puppeteer page context.
30. **3D content** -- Serve mode only, via Three.js.

---

## Architecture Notes

### Adding New Diagram Types

The established pattern from Mermaid integration:

1. **In `markdown-kit.js`**: Register a marked extension that converts ` ```lang ` code blocks into a `<div class="lang-diagram" data-source="...">` placeholder.
2. **In Puppeteer page context**: After the HTML is loaded, find all placeholders, call the library's render function, replace the div with the generated SVG.
3. **In serve mode**: Same as Puppeteer but runs in the live browser.

This pattern works for: D2, Nomnoml, WaveDrom, Railroad, Bytefield, Svgbob, ECharts, and any library that takes text input and produces SVG.

### Adding New Inline Extensions

For inline rendering (sparklines, QR codes, badges, progress bars):

1. Register a marked inline extension with a tokenizer regex.
2. The renderer directly generates the output HTML/SVG.
3. No Puppeteer-side processing needed.

### Server-Side vs Client-Side Rendering

Some libraries can render SVG purely in Node.js (no DOM needed):
- Nomnoml, WaveDrom, Bytefield, QR codes, Barcodes, Railroad diagrams, ECharts SSR

Others require a browser DOM context (Puppeteer page or serve mode):
- Mermaid (already handled), Markmap, D2 WASM, Excalidraw, abcjs, smiles-drawer, Cytoscape

The Puppeteer-based approach works for both categories since we already have a headless browser in the pipeline. For serve mode, everything runs in the real browser naturally.

### Kroki as Universal Fallback

Instead of bundling every WASM library, we could support a ` ```kroki-{format} ` syntax that sends diagram source to the Kroki API:

```markdown
\`\`\`kroki-plantuml
@startuml
Alice -> Bob: Hello
@enduml
\`\`\`
```

This covers 20+ formats with a single HTTP integration: PlantUML, Ditaa, Erd, BlockDiag, SeqDiag, ActDiag, NwDiag, PacketDiag, RackDiag, BPMN, Bytefield, C4, Excalidraw, Graphviz, Mermaid, Nomnoml, Structurizr, Svgbob, Symbolator, UMLet, Vega, Vega-Lite, WaveDrom, WireViz.

Requires network access. Could make Kroki endpoint configurable (default: `https://kroki.io`, override for self-hosted).

---

## What Mermaid Already Covers (No New Libraries Needed)

Our existing Mermaid integration (v11) already supports these diagram types. No new libraries are needed -- just ensure users know they are available:

| Mermaid Diagram | Use Case |
|-----------------|----------|
| Flowchart | Process flows, decision trees, algorithms |
| Sequence | API interactions, protocol flows |
| Class | OOP design, data models |
| State | State machines, lifecycle flows |
| ER | Database schemas |
| Gantt | Project timelines, scheduling |
| Pie | Simple proportional data |
| Journey | User experience flows |
| Gitgraph | Branch strategies, release flows |
| Mindmap | Brainstorming, topic hierarchies |
| Timeline | Chronological events |
| Sankey | Flow quantities, resource allocation |
| Quadrant | Priority matrices, risk assessment |
| C4 | Software architecture (Context, Container, Component, Code) |
| XY Chart | Line/bar charts with axes |
| Block | Block-based architecture diagrams |
| Architecture | Cloud/infrastructure diagrams with icons |
| Kanban | Task boards |
| Packet | Network packet structure |
| Radar | Multi-axis comparison |
| ZenUML | Sequence diagrams (alternative syntax) |

This means features like Timeline, Sankey, Kanban, Mindmap, and Org Charts are already covered without additional work. The value of adding standalone libraries (Markmap, vis-timeline, etc.) is only justified when they offer significantly better output quality or unique capabilities beyond Mermaid's version.

---

## Completed Features (March 2026)

Features implemented and removed from the backlog. The code is the documentation.

| Feature | Category | Completed |
|---------|----------|-----------|
| mhchem chemical formulas | Science | March 2026 -- KaTeX contrib loaded |
| Graphviz/DOT diagrams | Diagrams | March 2026 -- @hpcc-js/wasm-graphviz |
| Vega-Lite charts | Data Viz | March 2026 -- vega + vega-lite |
| Custom containers (:::) | Document | March 2026 -- custom marked extension |
| Variable substitution ({{key}}) | Document | March 2026 -- front matter interpolation |
| Mobile responsive serve mode | Serve | March 2026 -- viewport meta + responsive CSS |
| --mobile flag for phone PDFs | PDF | March 2026 |
| -o / --output custom path | CLI | March 2026 |
| Page numbers, headers/footers | PDF | March 2026 -- paged mode |
| PDF bookmarks / outline sidebar | PDF | March 2026 |
| PDF metadata from front matter | PDF | March 2026 |
| Clickable internal links | PDF | March 2026 -- TOC and footnote links |
| Front matter handling modes | Document | March 2026 -- strip/render/raw |

Previously confirmed as baseline capabilities (present before assessment):
Mermaid, KaTeX math, syntax highlighting, GFM tables, GFM alerts, footnotes, TOC, emoji, highlight, sub/sup, images, inline SVG.
