# Claude in Chrome Extension

> **Purpose**: Comprehensive reference for AI agents working with browser automation.
> Documents Anthropic's official Claude in Chrome extension - its tools, architecture,
> capabilities, and behavior patterns.

This document serves as the authoritative reference for understanding what the official
browser automation tool provides. See the [Selenium README](../README.md) for how our
custom implementation compares and where we aim to match or exceed these capabilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Complete Tool Reference](#complete-tool-reference)
   - [Session Management](#session-management)
   - [Navigation](#navigation)
   - [Content Extraction](#content-extraction)
   - [Interaction](#interaction)
   - [Debugging](#debugging)
   - [Utilities](#utilities)
3. [Architecture](#architecture)
4. [Security Model](#security-model)
5. [Technical Context](#technical-context)
6. [Industry Context](#industry-context)
7. [Appendices](#appendices)

---

## Overview

The Claude in Chrome extension is Anthropic's official browser automation tool, operating as an
MCP server that bridges Claude's AI capabilities with Chrome browser automation.

**Key Characteristics:**
- Official Anthropic product (currently in beta, requires Claude Max subscription)
- Runs as a Chrome extension with side panel interface
- Exposes 17 tools via MCP protocol
- Tab group-based session management
- Full accessibility tree access
- Screenshot and visual capabilities
- Network request monitoring
- Console log access

**Connection:**
- Requires Chrome browser with extension installed
- Extension must be running and connected to Claude Code CLI
- Use `/chrome` command in Claude Code to attempt reconnection
- Tab IDs are session-specific and cannot be reused across sessions

### Tool Summary (17 tools)

| Category        | Tool                                              | Description                         |
|-----------------|---------------------------------------------------|-------------------------------------|
| **Session**     | [`tabs_context_mcp`](#tabs_context_mcp)           | Get tab group context (call first)  |
|                 | [`tabs_create_mcp`](#tabs_create_mcp)             | Create new tab in group             |
| **Navigation**  | [`navigate`](#navigate)                           | Go to URL or back/forward           |
| **Content**     | [`get_page_text`](#get_page_text)                 | Extract all text (including hidden) |
|                 | [`read_page`](#read_page)                         | Get accessibility tree with ref IDs |
|                 | [`find`](#find)                                   | Find elements via natural language  |
| **Interaction** | [`computer`](#computer)                           | Unified mouse/keyboard/screenshot   |
|                 | [`form_input`](#form_input)                       | Set form values by ref ID           |
|                 | [`javascript_tool`](#javascript_tool)             | Execute JS in page context          |
| **Debugging**   | [`read_console_messages`](#read_console_messages) | Read browser console logs           |
|                 | [`read_network_requests`](#read_network_requests) | Read HTTP requests                  |
| **Utilities**   | [`resize_window`](#resize_window)                 | Set window dimensions               |
|                 | [`gif_creator`](#gif_creator)                     | Record interactions as GIF          |
|                 | [`upload_image`](#upload_image)                   | Upload screenshot to page           |
|                 | [`update_plan`](#update_plan)                     | Present plan for user approval      |
|                 | [`shortcuts_list`](#shortcuts_list)               | List available shortcuts            |
|                 | [`shortcuts_execute`](#shortcuts_execute)         | Run a shortcut/workflow             |

---

## Complete Tool Reference

### Session Management

##### `tabs_context_mcp`

Get context information about the current MCP tab group. **Must be called first** at the start
of each browser automation session.

```typescript
Parameters:
  createIfEmpty?: boolean  // Creates new MCP tab group if none exists

Returns:
  {
    availableTabs: [{tabId: number, title: string, url: string}],
    tabGroupId: number
  }
```

**Usage Notes:**
- Call this first to get valid tab IDs
- Each conversation should create its own new tab
- Tab IDs from previous sessions are invalid

---

##### `tabs_create_mcp`

Creates a new empty tab in the MCP tab group.

```typescript
Parameters: none

Returns: New tab information with tabId
```

**Usage Notes:**
- Use when you need a fresh tab for navigation
- Preferred over reusing existing tabs unless user explicitly requests it

---

#### Navigation

##### `navigate`

Navigate to a URL, or go forward/back in browser history.

```typescript
Parameters:
  url: string     // URL to navigate to, or "forward"/"back"
  tabId: number   // Tab ID to navigate

Returns:
  {title: string, url: string}
```

**Usage Notes:**
- URLs can be provided with or without protocol (defaults to https://)
- Handles redirects automatically
- Returns final URL after any redirects

---

#### Content Extraction

##### `get_page_text`

Extract raw text content from the page, prioritizing article content.

```typescript
Parameters:
  tabId: number   // Tab ID to extract text from

Returns: string   // Plain text content with metadata header
```

**Output Format:**
```
Title: Page Title
URL: https://example.com
Source element: <main>
---
[Extracted text content...]
```

**Critical Behavior - This is a KEY differentiator:**
- Extracts ALL text content including hidden/collapsed elements
- Uses `textContent` approach (not `innerText`)
- Includes content from modals, dialogs, and visually hidden elements
- Prioritizes main content area
- Normalizes whitespace

**Example:** On a Marriott page with hidden modal content:
```
Hertz President's Circle Status Hertz President's Circle Status
Marriott Bonvoy has partnered with Hertz to bring qualified members
complimentary Hertz Gold Plus Rewards President's Circle status.
Get the status you deserve. Register now. Close Dialog
```

Chrome captures both the visible label AND the hidden modal content.

---

##### `read_page`

Get an accessibility tree representation of elements on the page.

```typescript
Parameters:
  tabId: number           // Tab ID to read from
  depth?: number          // Max depth of tree traversal (default: 15)
  filter?: "interactive" | "all"  // Filter elements
  ref_id?: string         // Focus on specific element's subtree

Returns: Accessibility tree with reference IDs
```

**Output Format:**
```
ref_1: button "Submit"
  ref_2: text "Submit"
ref_3: input type="text" placeholder="Search"
ref_4: link "Home"
  ref_5: text "Home"
```

**Usage Notes:**
- Output limited to 50,000 characters
- Use smaller depth or ref_id to focus if output too large
- Reference IDs can be used with other tools (click, form_input)
- Includes visibility state and interactive element identification

---

##### `find`

Find elements on the page using natural language.

```typescript
Parameters:
  query: string    // Natural language description
  tabId: number    // Tab ID to search in

Returns: Up to 20 matching elements with reference IDs
```

**Usage Notes:**
- Uses semantic understanding to match elements
- Can search by purpose ("add to cart button") or content ("organic mango")
- More intuitive than CSS selectors for common use cases

---

#### Interaction

##### `computer`

Unified mouse and keyboard interaction tool.

```typescript
Parameters:
  action: "left_click" | "right_click" | "double_click" | "triple_click" |
          "type" | "key" | "screenshot" | "wait" | "scroll" | "scroll_to" |
          "left_click_drag" | "zoom" | "hover"
  tabId: number

  // For click actions:
  coordinate?: [number, number]  // [x, y] coordinates
  ref?: string                   // Element reference ID (alternative to coordinates)
  modifiers?: string             // "ctrl", "shift", "alt", "cmd"

  // For type/key actions:
  text?: string                  // Text to type or key to press
  repeat?: number                // Times to repeat key (1-100)

  // For wait action:
  duration?: number              // Seconds to wait (max 30)

  // For scroll action:
  scroll_direction?: "up" | "down" | "left" | "right"
  scroll_amount?: number         // Scroll wheel ticks (1-10)

  // For scroll_to action:
  ref?: string                   // Element to scroll into view

  // For zoom action:
  region?: [number, number, number, number]  // [x0, y0, x1, y1]

  // For left_click_drag:
  start_coordinate?: [number, number]
```

**Key Advantages:**
- Unified interface for all interactions
- Supports both coordinate-based and element-reference-based targeting
- Built-in visual verification via screenshots

---

##### `form_input`

Set values in form elements using element reference ID.

```typescript
Parameters:
  ref: string                      // Element reference from read_page
  value: string | boolean | number // Value to set
  tabId: number

Returns: Confirmation of value set
```

**Usage Notes:**
- Works with checkboxes (boolean), selects (option value/text), text inputs
- Uses reference IDs from read_page tool
- More reliable than click-based form filling

---

##### `javascript_tool`

Execute JavaScript code in the context of the current page.

```typescript
Parameters:
  action: "javascript_exec"  // Must be this value
  text: string               // JavaScript code to execute
  tabId: number

Returns: Result of the last expression
```

**Key Capability:**
- Code runs in page context with full DOM access
- Can interact with page variables and functions
- Useful for debugging and custom interactions
- Result of last expression is returned automatically
- Do NOT use `return` statements

**Example:**
```javascript
// Get text content comparison
const btn = document.querySelector('button');
({
  innerText: btn.innerText,
  textContent: btn.textContent
})
```

---

#### Debugging

##### `read_console_messages`

Read browser console messages (console.log, console.error, etc.).

```typescript
Parameters:
  tabId: number
  pattern?: string       // Regex pattern to filter messages
  onlyErrors?: boolean   // Only return error/exception messages
  limit?: number         // Max messages (default: 100)
  clear?: boolean        // Clear messages after reading

Returns: Console messages with type, content, timestamp
```

**Key Capability:**
- Essential for debugging page issues
- Can identify JavaScript errors affecting functionality
- ALWAYS provide a pattern to avoid overwhelming output

---

##### `read_network_requests`

Read HTTP network requests (XHR, Fetch, documents, images, etc.).

```typescript
Parameters:
  tabId: number
  urlPattern?: string    // Filter requests by URL substring
  limit?: number         // Max requests (default: 100)
  clear?: boolean        // Clear requests after reading

Returns: Network requests with URL, method, status, timing
```

**Usage Notes:**
- Includes cross-origin requests
- Auto-cleared on domain navigation
- Use urlPattern to filter (e.g., "/api/")

---

#### Utilities

##### `resize_window`

Resize the browser window to specified dimensions.

```typescript
Parameters:
  width: number    // Target width in pixels
  height: number   // Target height in pixels
  tabId: number

Returns: Confirmation with new dimensions
```

**Usage Notes:**
- Useful for responsive design testing

---

##### `gif_creator`

Record browser interactions and export as animated GIF.

```typescript
Parameters:
  action: "start_recording" | "stop_recording" | "export" | "clear"
  tabId: number
  download?: boolean      // Set true for export to download GIF
  filename?: string       // Name for exported GIF
  options?: {
    showClickIndicators?: boolean  // Orange circles at clicks
    showDragPaths?: boolean        // Red arrows for drags
    showActionLabels?: boolean     // Black labels for actions
    showProgressBar?: boolean      // Orange progress bar
    showWatermark?: boolean        // Claude logo watermark
    quality?: number               // 1-30 (lower = better)
  }
```

**Key Capability:**
- Essential for creating visual demonstrations
- Useful for documenting multi-step processes

**Usage Pattern:**
1. Start recording
2. Take screenshot (initial state)
3. Perform interactions
4. Take screenshot (final state)
5. Stop recording
6. Export with download=true

---

##### `upload_image`

Upload a screenshot or user-uploaded image to file input or drag target.

```typescript
Parameters:
  imageId: string          // ID of captured screenshot
  tabId: number
  ref?: string             // Element reference for file inputs
  coordinate?: [number, number]  // For drag & drop targets
  filename?: string        // Filename for upload
```

---

##### `update_plan`

Present a plan to the user for approval before taking actions.

```typescript
Parameters:
  domains: string[]     // List of domains to visit
  approach: string[]    // High-level description of steps

Returns: User approval status
```

---

##### `shortcuts_list`

List all available shortcuts and workflows.

```typescript
Parameters:
  tabId: number
Returns: List of shortcuts with commands and descriptions
```

---

##### `shortcuts_execute`

Execute a shortcut or workflow.

```typescript
Parameters:
  tabId: number
  command?: string      // Command name (without leading slash)
  shortcutId?: string   // Shortcut ID
```

---

## Architecture

### Tab Group Model

The Chrome extension uses a tab group-based session model:

```
Tab Group (ID: 223543734)
├── Tab 1 (ID: 1503493187) - "Marriott Page"
├── Tab 2 (ID: 1503493202) - "New Tab"
└── Tab 3 (ID: 1503493215) - "Google Search"
```

- **Tab Groups**: All tabs for a Claude session are grouped together
- **Tab IDs**: Each tab has a unique numeric ID (session-specific)
- **Session Isolation**: Each conversation gets its own tab group
- **Persistence**: Tab state persists across tool calls within a session

### MCP Communication

```
Claude Code CLI <--MCP Protocol--> Chrome Extension <--Chrome APIs--> Browser
```

The extension acts as an MCP server, receiving tool calls from Claude Code and translating them
into Chrome extension API calls.

### Chrome Security Model

The Chrome extension implements strict security rules:

- **Injection Defense**: Protection against malicious web content instructions
- **Prohibited Actions**: Financial data, account creation, permanent deletions
- **Explicit Permission**: Required for purchases, sharing, publishing, sending
- **Cookie Consent**: Automatically declines cookies unless instructed otherwise
- **Copyright Protection**: Never reproduces large chunks of copyrighted content

**Known Vulnerability**: ~11.2% success rate for prompt injection attacks in controlled testing
(Anthropic's own security research). Web content can potentially manipulate Claude's actions.

---

## Technical Context

### Text Extraction: innerText vs textContent

The fundamental difference causing the text extraction gap:

| Property | Behavior | Performance |
|----------|----------|-------------|
| `innerText` | Visible text only, layout-aware | Slow (triggers reflow) |
| `textContent` | All text including hidden | Fast (no layout needed) |

**HTML Example:**
```html
<button>
  <span>Visible Label</span>
  <div class="d-none">Hidden modal content with details...</div>
</button>
```

**Results:**
- `button.innerText` → "Visible Label"
- `button.textContent` → "Visible Label Hidden modal content with details..."

Chrome uses `textContent`, which captures all text including hidden elements.

### Bootstrap d-none Class

Many modern sites use Bootstrap's `d-none` class for hidden content:

```css
.d-none { display: none !important; }
```

This hidden content is often:
- Modal dialog bodies
- Collapsed accordion content
- Tooltip text
- Screen reader content

Chrome's `get_page_text` captures all of this hidden content.

### ARIA Accessibility Trees

Chrome's `read_page` provides accessibility tree access with actionable reference IDs:

```
ref_1: button "Submit"
  ref_2: text "Submit"
ref_3: input type="text" placeholder="Search"
ref_4: link "Home"
  ref_5: text "Home"
```

Reference IDs (e.g., `ref_1`) can be used directly with `computer` and `form_input` tools for
element interaction without needing CSS selectors.

---

## Industry Context

### Browser Automation Landscape (2024-2025)

Based on research from Stack Overflow Developer Survey 2025 and industry analysis:

#### Market Leaders

| Tool | Strength | Weakness |
|------|----------|----------|
| Playwright | Speed, modern API, cross-browser | Learning curve |
| Selenium | Language support, legacy compatibility | Maintenance burden, slower |
| Puppeteer | Chrome integration, DevTools access | Chrome-only |
| Skyvern | AI-powered, handles unknown sites | Cost, complexity |
| Browserbase | Managed infrastructure, observability | Vendor dependency |

#### AI Browser Automation

| Tool | Approach | Performance |
|------|----------|-------------|
| Claude in Chrome | Extension + MCP | Good for simple tasks, 11% injection vulnerability |
| OpenAI Operator | CUA model + browser | State-of-art on benchmarks |
| Gemini Computer Use | Visual understanding | Lowest latency |
| Skyvern | LLM + computer vision | 85.8% on WebVoyager |

#### User Pain Points (from research)

1. **Session Persistence**: Re-authentication consumes up to 70% of runtime
2. **CAPTCHA Solving**: Modern CAPTCHAs detect non-human patterns
3. **Complex Workflows**: 66% frustrated with "almost right" AI solutions
4. **Maintenance Overhead**: Selector brittleness, layout changes
5. **Error Recovery**: Lack of sophisticated retry/escalation logic

#### Key Insights

- **textContent vs innerText**: Performance difference of 300ms vs 1ms in benchmarks
- **ARIA-based locators**: More stable than CSS selectors across layout changes
- **Session persistence**: Can reduce setup time by 90%
- **MCP security**: 7-10% of MCP servers have documented vulnerabilities

---

## Appendices

### Appendix A: Test Pages

Use these pages for consistent comparison testing:

1. **Marriott Bonvoy Status Pages** (hidden modal content):
   - https://www.marriott.com/loyalty/member-benefits/silver.mi
   - https://www.marriott.com/loyalty/member-benefits/gold.mi
   - https://www.marriott.com/loyalty/member-benefits/platinum.mi
   - https://www.marriott.com/loyalty/member-benefits/titanium.mi
   - https://www.marriott.com/loyalty/member-benefits/ambassador.mi

2. **Example.com** (simple, stable):
   - https://example.com

3. **MDN Web Docs** (complex structure):
   - https://developer.mozilla.org/

### Appendix B: Chrome Extension Parameters (Complete)

```typescript
// tabs_context_mcp
{createIfEmpty?: boolean}

// tabs_create_mcp
{}

// navigate
{url: string, tabId: number}

// get_page_text
{tabId: number}

// read_page
{tabId: number, depth?: number, filter?: "interactive" | "all", ref_id?: string}

// find
{query: string, tabId: number}

// computer
{
  action: "left_click" | "right_click" | "double_click" | "triple_click" |
          "type" | "key" | "screenshot" | "wait" | "scroll" | "scroll_to" |
          "left_click_drag" | "zoom" | "hover",
  tabId: number,
  coordinate?: [number, number],
  ref?: string,
  text?: string,
  duration?: number,
  scroll_direction?: "up" | "down" | "left" | "right",
  scroll_amount?: number,
  region?: [number, number, number, number],
  modifiers?: string,
  repeat?: number,
  start_coordinate?: [number, number]
}

// form_input
{ref: string, value: string | boolean | number, tabId: number}

// javascript_tool
{action: "javascript_exec", text: string, tabId: number}

// read_console_messages
{tabId: number, pattern?: string, onlyErrors?: boolean, limit?: number, clear?: boolean}

// read_network_requests
{tabId: number, urlPattern?: string, limit?: number, clear?: boolean}

// resize_window
{width: number, height: number, tabId: number}

// gif_creator
{
  action: "start_recording" | "stop_recording" | "export" | "clear",
  tabId: number,
  download?: boolean,
  filename?: string,
  options?: {
    showClickIndicators?: boolean,
    showDragPaths?: boolean,
    showActionLabels?: boolean,
    showProgressBar?: boolean,
    showWatermark?: boolean,
    quality?: number
  }
}

// upload_image
{imageId: string, tabId: number, ref?: string, coordinate?: [number, number], filename?: string}

// update_plan
{domains: string[], approach: string[]}

// shortcuts_list
{tabId: number}

// shortcuts_execute
{tabId: number, command?: string, shortcutId?: string}
```

### Appendix C: Glossary

| Term | Definition |
|------|------------|
| ARIA | Accessible Rich Internet Applications - web accessibility standard |
| CDP | Chrome DevTools Protocol - low-level Chrome control API |
| HAR | HTTP Archive - standard format for network traffic logs |
| LCP | Largest Contentful Paint - Core Web Vital metric |
| CLS | Cumulative Layout Shift - Core Web Vital metric |
| INP | Interaction to Next Paint - Core Web Vital metric |
| MCP | Model Context Protocol - Anthropic's AI tool protocol |
| innerText | DOM property returning visible text only |
| textContent | DOM property returning all text including hidden |
| d-none | Bootstrap CSS class for display: none |

### Appendix D: Official MCP Tool Definitions (Verbatim)

These are the exact tool definitions from the Claude in Chrome MCP server. Preserved verbatim
for tracking changes over time. Last verified: 2024-12-22.

---

#### `tabs_context_mcp`

**Full name:** `mcp__claude-in-chrome__tabs_context_mcp`

**Description:**
Get context information about the current MCP tab group. Returns all tab IDs inside the group if it exists. CRITICAL: You must get the context at least once before using other browser automation tools so you know what tabs exist. Each new conversation should create its own new tab (using tabs_create_mcp) rather than reusing existing tabs, unless the user explicitly asks to use an existing tab.

**Parameters:**
- `createIfEmpty` (optional): boolean - Creates a new MCP tab group if none exists, creates a new Window with a new tab group containing an empty tab (which can be used for this conversation). If a MCP tab group already exists, this parameter has no effect.

---

#### `tabs_create_mcp`

**Full name:** `mcp__claude-in-chrome__tabs_create_mcp`

**Description:**
Creates a new empty tab in the MCP tab group. CRITICAL: You must get the context using tabs_context_mcp at least once before using other browser automation tools so you know what tabs exist.

**Parameters:** None

---

#### `navigate`

**Full name:** `mcp__claude-in-chrome__navigate`

**Description:**
Navigate to a URL, or go forward/back in browser history. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `url` (required): string - The URL to navigate to. Can be provided with or without protocol (defaults to https://). Use "forward" to go forward in history or "back" to go back in history.
- `tabId` (required): number - Tab ID to navigate. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.

---

#### `get_page_text`

**Full name:** `mcp__claude-in-chrome__get_page_text`

**Description:**
Extract raw text content from the page, prioritizing article content. Ideal for reading articles, blog posts, or other text-heavy pages. Returns plain text without HTML formatting. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `tabId` (required): number - Tab ID to extract text from. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.

---

#### `read_page`

**Full name:** `mcp__claude-in-chrome__read_page`

**Description:**
Get an accessibility tree representation of elements on the page. By default returns all elements including non-visible ones. Output is limited to 50000 characters. If the output exceeds this limit, you will receive an error asking you to specify a smaller depth or focus on a specific element using ref_id. Optionally filter for only interactive elements. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `tabId` (required): number - Tab ID to read from. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.
- `depth` (optional): number - Maximum depth of the tree to traverse (default: 15). Use a smaller depth if output is too large.
- `filter` (optional): "interactive" | "all" - Filter elements: "interactive" for buttons/links/inputs only, "all" for all elements including non-visible ones (default: all elements)
- `ref_id` (optional): string - Reference ID of a parent element to read. Will return the specified element and all its children. Use this to focus on a specific part of the page when output is too large.

---

#### `find`

**Full name:** `mcp__claude-in-chrome__find`

**Description:**
Find elements on the page using natural language. Can search for elements by their purpose (e.g., "search bar", "login button") or by text content (e.g., "organic mango product"). Returns up to 20 matching elements with references that can be used with other tools. If more than 20 matches exist, you'll be notified to use a more specific query. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `query` (required): string - Natural language description of what to find (e.g., "search bar", "add to cart button", "product title containing organic")
- `tabId` (required): number - Tab ID to search in. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.

---

#### `computer`

**Full name:** `mcp__claude-in-chrome__computer`

**Description:**
Use a mouse and keyboard to interact with a web browser, and take screenshots. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.
* Whenever you intend to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your click location so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.

**Parameters:**
- `action` (required): string - The action to perform:
  * `left_click`: Click the left mouse button at the specified coordinates.
  * `right_click`: Click the right mouse button at the specified coordinates to open context menus.
  * `double_click`: Double-click the left mouse button at the specified coordinates.
  * `triple_click`: Triple-click the left mouse button at the specified coordinates.
  * `type`: Type a string of text.
  * `screenshot`: Take a screenshot of the screen.
  * `wait`: Wait for a specified number of seconds.
  * `scroll`: Scroll up, down, left, or right at the specified coordinates.
  * `key`: Press a specific keyboard key.
  * `left_click_drag`: Drag from start_coordinate to coordinate.
  * `zoom`: Take a screenshot of a specific region for closer inspection.
  * `scroll_to`: Scroll an element into view using its element reference ID from read_page or find tools.
  * `hover`: Move the mouse cursor to the specified coordinates or element without clicking. Useful for revealing tooltips, dropdown menus, or triggering hover states.
- `tabId` (required): number - Tab ID to execute the action on. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.
- `coordinate` (optional): [number, number] - (x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required for `left_click`, `right_click`, `double_click`, `triple_click`, and `scroll`. For `left_click_drag`, this is the end position.
- `ref` (optional): string - Element reference ID from read_page or find tools (e.g., "ref_1", "ref_2"). Required for `scroll_to` action. Can be used as alternative to `coordinate` for click actions.
- `text` (optional): string - The text to type (for `type` action) or the key(s) to press (for `key` action). For `key` action: Provide space-separated keys (e.g., "Backspace Backspace Delete"). Supports keyboard shortcuts using the platform's modifier key (use "cmd" on Mac, "ctrl" on Windows/Linux, e.g., "cmd+a" or "ctrl+a" for select all).
- `duration` (optional): number - The number of seconds to wait. Required for `wait`. Minimum 0, maximum 30 seconds.
- `scroll_direction` (optional): "up" | "down" | "left" | "right" - The direction to scroll. Required for `scroll`.
- `scroll_amount` (optional): number - The number of scroll wheel ticks. Optional for `scroll`, defaults to 3. Minimum 1, maximum 10.
- `region` (optional): [number, number, number, number] - (x0, y0, x1, y1): The rectangular region to capture for `zoom`. Coordinates define a rectangle from top-left (x0, y0) to bottom-right (x1, y1) in pixels from the viewport origin. Required for `zoom` action. Useful for inspecting small UI elements like icons, buttons, or text.
- `modifiers` (optional): string - Modifier keys for click actions. Supports: "ctrl", "shift", "alt", "cmd" (or "meta"), "win" (or "windows"). Can be combined with "+" (e.g., "ctrl+shift", "cmd+alt").
- `repeat` (optional): number - Number of times to repeat the key sequence. Only applicable for `key` action. Must be a positive integer between 1 and 100. Default is 1. Useful for navigation tasks like pressing arrow keys multiple times.
- `start_coordinate` (optional): [number, number] - (x, y): The starting coordinates for `left_click_drag`.

---

#### `form_input`

**Full name:** `mcp__claude-in-chrome__form_input`

**Description:**
Set values in form elements using element reference ID from the read_page tool. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `ref` (required): string - Element reference ID from the read_page tool (e.g., "ref_1", "ref_2")
- `value` (required): string | boolean | number - The value to set. For checkboxes use boolean, for selects use option value or text, for other inputs use appropriate string/number
- `tabId` (required): number - Tab ID to set form value in. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.

---

#### `javascript_tool`

**Full name:** `mcp__claude-in-chrome__javascript_tool`

**Description:**
Execute JavaScript code in the context of the current page. The code runs in the page's context and can interact with the DOM, window object, and page variables. Returns the result of the last expression or any thrown errors. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `action` (required): string - Must be set to 'javascript_exec'
- `text` (required): string - The JavaScript code to execute. The code will be evaluated in the page context. The result of the last expression will be returned automatically. Do NOT use 'return' statements - just write the expression you want to evaluate (e.g., 'window.myData.value' not 'return window.myData.value'). You can access and modify the DOM, call page functions, and interact with page variables.
- `tabId` (required): number - Tab ID to execute the code in. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.

---

#### `read_console_messages`

**Full name:** `mcp__claude-in-chrome__read_console_messages`

**Description:**
Read browser console messages (console.log, console.error, console.warn, etc.) from a specific tab. Useful for debugging JavaScript errors, viewing application logs, or understanding what's happening in the browser console. Returns console messages from the current domain only. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs. IMPORTANT: Always provide a pattern to filter messages - without a pattern, you may get too many irrelevant messages.

**Parameters:**
- `tabId` (required): number - Tab ID to read console messages from. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.
- `pattern` (optional): string - Regex pattern to filter console messages. Only messages matching this pattern will be returned (e.g., 'error|warning' to find errors and warnings, 'MyApp' to filter app-specific logs). You should always provide a pattern to avoid getting too many irrelevant messages.
- `onlyErrors` (optional): boolean - If true, only return error and exception messages. Default is false (return all message types).
- `limit` (optional): number - Maximum number of messages to return. Defaults to 100. Increase only if you need more results.
- `clear` (optional): boolean - If true, clear the console messages after reading to avoid duplicates on subsequent calls. Default is false.

---

#### `read_network_requests`

**Full name:** `mcp__claude-in-chrome__read_network_requests`

**Description:**
Read HTTP network requests (XHR, Fetch, documents, images, etc.) from a specific tab. Useful for debugging API calls, monitoring network activity, or understanding what requests a page is making. Returns all network requests made by the current page, including cross-origin requests. Requests are automatically cleared when the page navigates to a different domain. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `tabId` (required): number - Tab ID to read network requests from. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.
- `urlPattern` (optional): string - Optional URL pattern to filter requests. Only requests whose URL contains this string will be returned (e.g., '/api/' to filter API calls, 'example.com' to filter by domain).
- `limit` (optional): number - Maximum number of requests to return. Defaults to 100. Increase only if you need more results.
- `clear` (optional): boolean - If true, clear the network requests after reading to avoid duplicates on subsequent calls. Default is false.

---

#### `resize_window`

**Full name:** `mcp__claude-in-chrome__resize_window`

**Description:**
Resize the current browser window to specified dimensions. Useful for testing responsive designs or setting up specific screen sizes. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.

**Parameters:**
- `width` (required): number - Target window width in pixels
- `height` (required): number - Target window height in pixels
- `tabId` (required): number - Tab ID to get the window for. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.

---

#### `gif_creator`

**Full name:** `mcp__claude-in-chrome__gif_creator`

**Description:**
Manage GIF recording and export for browser automation sessions. Control when to start/stop recording browser actions (clicks, scrolls, navigation), then export as an animated GIF with visual overlays (click indicators, action labels, progress bar, watermark). All operations are scoped to the tab's group. When starting recording, take a screenshot immediately after to capture the initial state as the first frame. When stopping recording, take a screenshot immediately before to capture the final state as the last frame. For export, either provide 'coordinate' to drag/drop upload to a page element, or set 'download: true' to download the GIF.

**Parameters:**
- `action` (required): string - Action to perform: 'start_recording' (begin capturing), 'stop_recording' (stop capturing but keep frames), 'export' (generate and export GIF), 'clear' (discard frames)
- `tabId` (required): number - Tab ID to identify which tab group this operation applies to
- `download` (optional): boolean - Always set this to true for the 'export' action only. This causes the gif to be downloaded in the browser.
- `filename` (optional): string - Optional filename for exported GIF (default: 'recording-[timestamp].gif'). For 'export' action only.
- `options` (optional): object - Optional GIF enhancement options for 'export' action:
  - `showClickIndicators` (boolean): Show orange circles at click locations (default: true)
  - `showDragPaths` (boolean): Show red arrows for drag actions (default: true)
  - `showActionLabels` (boolean): Show black labels describing actions (default: true)
  - `showProgressBar` (boolean): Show orange progress bar at bottom (default: true)
  - `showWatermark` (boolean): Show Claude logo watermark (default: true)
  - `quality` (number): GIF compression quality, 1-30 (lower = better quality, slower encoding). Default: 10

---

#### `upload_image`

**Full name:** `mcp__claude-in-chrome__upload_image`

**Description:**
Upload a previously captured screenshot or user-uploaded image to a file input or drag & drop target. Supports two approaches: (1) ref - for targeting specific elements, especially hidden file inputs, (2) coordinate - for drag & drop to visible locations like Google Docs. Provide either ref or coordinate, not both.

**Parameters:**
- `imageId` (required): string - ID of a previously captured screenshot (from the computer tool's screenshot action) or a user-uploaded image
- `tabId` (required): number - Tab ID where the target element is located. This is where the image will be uploaded to.
- `ref` (optional): string - Element reference ID from read_page or find tools (e.g., "ref_1", "ref_2"). Use this for file inputs (especially hidden ones) or specific elements. Provide either ref or coordinate, not both.
- `coordinate` (optional): [number, number] - Viewport coordinates [x, y] for drag & drop to a visible location. Use this for drag & drop targets like Google Docs. Provide either ref or coordinate, not both.
- `filename` (optional): string - Optional filename for the uploaded file (default: "image.png")

---

#### `update_plan`

**Full name:** `mcp__claude-in-chrome__update_plan`

**Description:**
Present a plan to the user for approval before taking actions. The user will see the domains you intend to visit and your approach. Once approved, you can proceed with actions on the approved domains without additional permission prompts.

**Parameters:**
- `domains` (required): string[] - List of domains you will visit (e.g., ['github.com', 'stackoverflow.com']). These domains will be approved for the session when the user accepts the plan.
- `approach` (required): string[] - High-level description of what you will do. Focus on outcomes and key actions, not implementation details. Be concise - aim for 3-7 items.

---

#### `shortcuts_list`

**Full name:** `mcp__claude-in-chrome__shortcuts_list`

**Description:**
List all available shortcuts and workflows (shortcuts and workflows are interchangeable). Returns shortcuts with their commands, descriptions, and whether they are workflows. Use shortcuts_execute to run a shortcut or workflow.

**Parameters:**
- `tabId` (required): number - Tab ID to list shortcuts from. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.

---

#### `shortcuts_execute`

**Full name:** `mcp__claude-in-chrome__shortcuts_execute`

**Description:**
Execute a shortcut or workflow by running it in a new sidepanel window using the current tab (shortcuts and workflows are interchangeable). Use shortcuts_list first to see available shortcuts. This starts the execution and returns immediately - it does not wait for completion.

**Parameters:**
- `tabId` (required): number - Tab ID to execute the shortcut on. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID.
- `command` (optional): string - The command name of the shortcut to execute (e.g., 'debug', 'summarize'). Do not include the leading slash.
- `shortcutId` (optional): string - The ID of the shortcut to execute

---

## Changelog

| Date | Change |
|------|--------|
| 2024-12-22 | Initial documentation created |

---

## Contributing

When updating this document:

1. Keep tool reference current as the extension evolves
2. Add new test pages as edge cases are discovered
3. Update technical context with new findings
4. Maintain changelog with significant updates