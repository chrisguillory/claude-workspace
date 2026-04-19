# imessage-kit

MCP server for reading, searching, and sending iMessages via macOS `chat.db`.

Combines the best of five MIT-licensed implementations into a single server that handles the `attributedBody` binary format (93% of modern messages), edited message history, HEIC image conversion, and contact resolution — things no existing tool gets fully right.

## Prerequisites

**Full Disk Access** is required for reading `~/Library/Messages/chat.db`.

Grant it to the application that launches Claude Code:

```bash
# Find which app needs FDA
imessage-kit diagnose
```

The `diagnose` command walks the process tree and reports exactly which `.app` to add. Then:

1. **System Settings → Privacy & Security → Full Disk Access**
2. Add the reported application (e.g., `/Applications/iTerm.app`)
3. Restart Claude Code

## Installation

### Quick Start

```bash
uv tool install git+https://github.com/chrisguillory/claude-workspace.git#subdirectory=mcp/imessage-kit
```

This installs two commands to `~/.local/bin/`:
- `imessage-kit-mcp` — MCP server (used by Claude Code)
- `imessage-kit` — CLI client

### Configure Claude Code

```bash
claude mcp add --scope user imessage-kit -- imessage-kit-mcp
```

### Upgrade

```bash
uv tool upgrade imessage-kit
```

### Local Development

Editable install (changes take effect immediately):

```bash
uv tool install --editable /path/to/claude-workspace/mcp/imessage-kit
claude mcp add --scope user imessage-kit -- imessage-kit-mcp
```

## Tools

### `list_chats`

List chats ordered by most recent message date. Includes participant names, last message preview, unread count, and message/attachment counts.

Filters: `is_group`, `active_since`, `handle`. Pagination: offset/limit.

### `get_messages`

Get messages from a chat by handle (phone/email) or chat_id. Parses `attributedBody` for messages with NULL text column. Includes attachment metadata and reactions inline.

Cursor pagination via `before_rowid`/`after_rowid`. Filters: `date_from`, `date_to`, `has_attachment`, `from_me`.

### `search_messages`

Full-text search across all chats. Two-pass: SQL LIKE on `text` column, then batch-parses `attributedBody` blobs for messages with NULL text (capped at 50K row scan).

Filters: `handle`, `date_from`, `date_to`, `has_attachment`, `from_me`.

### `list_attachments`

List attachment metadata for a chat (type, size, filename, availability). Use `get_attachment` with the returned `attachment_id` to retrieve content.

Filters: `mime_type` (prefix match, e.g., `image/`), `date_from`, `date_to`, `from_me`.

### `get_attachment`

Retrieve an attachment by ID. Requires `mode`:

- **`view`** — Base64-encoded image for Claude vision. HEIC auto-converted to JPEG via macOS `sips`. 10MB size cap.
- **`save`** — Native file copied to temp dir (no conversion). Returns the file path.

### `send_message`

Send a message to a handle (1:1) or chat GUID (group) via AppleScript. Two-step UX: `confirm=false` previews the message, `confirm=true` sends it. Security boundary is `ToolAnnotations(destructiveHint=True)`.

`service_type`: `auto` (Messages.app decides), `iMessage`, or `SMS`.

### `lookup_contact`

Search macOS AddressBook by name (fuzzy via thefuzz), phone, or email. Returns all handles for matched contacts — use a handle from the results with `get_messages`, `send_message`, etc.

### `get_unread`

Get unread incoming messages across all chats. Excludes tapback reactions. Cross-device read sync may cause false positives.

### `diagnose`

Health check: Full Disk Access status, DB connectivity, message count, contacts accessibility, macOS version. Always works — even without FDA, reports exactly what's wrong and which app needs the permission.

### `get_chat_thread`

Get an inline reply thread within a chat by `thread_originator_guid` (available in `get_messages` output).

### `get_chat_info`

Detailed chat metadata: participants, group name, service type, message/attachment counts, date range.

## Data Model

imessage-kit uses Apple's own terminology from `chat.db`:

| Term           | What it is                                                                       |
|----------------|----------------------------------------------------------------------------------|
| **handle**     | A phone number or email — a communication endpoint, not a person                 |
| **chat**       | A conversation thread with one or more handles                                   |
| **message**    | A single sent or received message                                                |
| **attachment** | A file (photo, PDF, video) linked to a message                                   |
| **contact**    | A person from macOS AddressBook (separate from chat.db) — resolved at query time |

A person with a phone AND email has two separate chats. `lookup_contact` returns all handles for a person; use a specific handle with other tools.

## Limitations

- **Reactions**: Can read tapback reactions on messages but cannot send them. macOS does not expose a scripting API for tapbacks.
- **Delivery confirmation**: `send_message` returns success when Messages.app accepts the command, not when the message is delivered. "Not Delivered" errors are asynchronous and not reported.
- **iCloud-offloaded attachments**: Files may exist in the database but not on disk if iCloud storage optimization offloaded them. `get_attachment` reports this; open Messages.app to trigger download.
- **Group chat naming**: Can read group names but cannot set them (read-only in AppleScript dictionary).

## Attribution

Built from MIT-licensed implementations:

| Component                                   | Source                                                                                                     |
|---------------------------------------------|------------------------------------------------------------------------------------------------------------|
| `attributedBody` parser (dynamic 0x2b scan) | [tszaks/multimodal-imessage-mcp](https://github.com/tszaks/multimodal-imessage-mcp) (MIT)                  |
| LE variable-length integer decoding         | [carterlasalle/mac_messages_mcp](https://github.com/carterlasalle/mac_messages_mcp) (MIT)                  |
| NSMutableString fallback + U+FFFC filtering | [anipotts/imessage-mcp](https://github.com/anipotts/imessage-mcp) (MIT)                                    |
| Contact resolution via AddressBook          | [carterlasalle/mac_messages_mcp](https://github.com/carterlasalle/mac_messages_mcp) (MIT)                  |
| Schema knowledge (edits, tapbacks, groups)  | [ReagentX/imessage-exporter](https://github.com/ReagentX/imessage-exporter) (GPL-3.0, approach study only) |