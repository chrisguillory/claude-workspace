"""Pydantic models for Claude Code session JSONL records.

This module defines strict types for all record types found in Claude Code session files.
Uses discriminated unions for type-safe parsing of heterogeneous JSONL data.

CLAUDE CODE VERSION COMPATIBILITY:
- Schema v0.1.1: Added todos field for Claude Code 2.0.47+
- Schema v0.1.2: Added error field to AssistantRecord for Claude Code 2.0.49+
- Schema v0.1.3: Added slug, DocumentContent, ContextManagement, SkillToolInput,
                 image/jpeg support, ECONNRESET error code for Claude Code 2.0.51+
- Schema v0.1.4: Added EnterPlanMode tool, slug to CompactBoundarySystemRecord for 2.0.62+
- Schema v0.1.5: Added AgentOutputTool input model for Claude Code 2.0.64+
- Schema v0.1.6: Added TaskOutput tool input model for Claude Code 2.0.65+
- Schema v0.1.7: Added model_context_window_exceeded stop_reason for context overflow handling
- Schema v0.1.8: Added sourceToolUseID, EmptyError, BeforeValidator for ultrathink case normalization (2.0.76+)
- Schema v0.1.9: Added CustomTitleRecord for user-defined session names
- Schema v0.2.0: Added sourceToolAssistantUUID to UserRecord, TurnDurationSystemRecord for Claude Code 2.1.1+
- Schema v0.2.4: Added ProgressRecord (hook/mcp/bash/task progress), MicrocompactBoundarySystemRecord,
                 allowedPrompts to ExitPlanModeToolInput, fixed MCPSearchToolInput max_results type (2.1.9+)
- Schema v0.2.5: Added permissionMode to UserRecord, apiError to AssistantRecord (2.1.15+)
- Schema v0.2.6: Added StopHookSummarySystemRecord, resume field to AgentProgressData (2.1.14+),
                 TaskCreate/TaskUpdate/TaskList tool inputs and results (2.1.17+)
- Schema v0.2.7: Added SimpleThinkingMetadata, McpMeta, MCPStructuredContent, Task tool mode field (2.1.19+)
- Schema v0.2.8: Added timeoutMs to BashProgressData (2.1.25+)
- Schema v0.2.9: Added inference_geo to TokenUsage, claude-opus-4-6 model ID,
                 PrLinkRecord, SavedHookContextRecord (2.1.32+)
- Schema v0.2.10: Added iterations to TokenUsage, BashOutputToolResult, ReadMcpResourceToolResult,
                  noOutputExpected to BashToolResult, 'killed' status, addBlocks to TaskUpdateToolInput,
                  command to TaskStopToolResult, appliedOffset to GrepToolResult, AgentState (2.1.38+)
- Schema v0.2.11: Added speed to TokenUsage, pages to ReadToolInput (2.1.41+)
- Schema v0.2.12: Added totalBytes/taskId to BashProgressData, persistedOutputPath/persistedOutputSize
                  to BashToolResult, annotations to AskUserQuestionToolResult (2.1.45+)
- Schema v0.2.13: Added claude-sonnet-4-6 model ID, max_tokens stop_reason,
                  canReadOutputFile to AsyncTaskLaunchResult (2.1.47+)
- Schema v0.2.14: Added BridgeStatusSystemRecord for /remote-control feature,
                  TaskGetToolResult, markdown to QuestionOption,
                  pending_mcp_servers to ToolSearchToolResult (2.1.51+)
- Schema v0.2.15: Added teamName/agentName to all record types, EnterWorktreeToolInput,
                  TeamCreateToolInput, SendMessageToolInput, Agent tool team_name/name fields,
                  migrated services from SessionRecordAdapter to validate_session_record (2.1.63+)
- Schema v0.2.16: Added LastPromptRecord, made AgentProgressData.normalizedMessages optional (2.1.69+)
- Schema v0.2.17: Added promptId to UserRecord, agentId to LocalCommand/Microcompact/TurnDuration
                 system records, isolation to TaskToolInput (Agent tool), preview to QuestionOption,
                 made TaskCreateToolInput.activeForm optional (2.1.74+)
- Schema v0.2.18: Added entrypoint field to all record types (from CLAUDE_CODE_ENTRYPOINT env var;
                 known values: cli, sdk-cli, sdk-ts, sdk-py, mcp, claude-vscode, claude-desktop,
                 claude-code-github-action, local-agent, remote), made AgentToolInput.subagent_type
                 optional (2.1.80+)
- Schema v0.2.19: Added agentId to CompactBoundary/ApiError system records (agent sidechain files)
- Schema v0.2.20: Added AttachmentRecord, PermissionModeRecord, WorktreeStateRecord,
                  upgradeNudge to BridgeStatusSystemRecord, budgetTokens/skills/mcp to UserRecord (2.1.90+)
- Schema v0.2.21: Added ScheduledTaskFireSystemRecord (/loop, CronCreate 2.1.85+),
                  CronCreate/CronDelete/CronList tool inputs, SendMessageSimpleToolInput (2.1.81+),
                  CompactMetadata.preCompactDiscoveredTools, expanded PermissionModeRecord permissionMode,
                  agentId to InformationalSystemRecord/AttachmentRecord, slug to InformationalSystemRecord,
                  Task.activeForm optional, Grep -r flag, fixed system dispatch for unknown subtypes (2.1.92+)
- Schema v0.2.22: Added claude-opus-4-7 model ID (2.1.111 Opus 4.7 launch), UsageIteration for
                  TokenUsage.iterations (2.1.100+), AwaySummarySystemRecord for /recap feature (2.1.108+),
                  WorktreeSessionData.enteredExisting with optional originalBranch/originalHeadCommit
                  (2.1.105 EnterWorktree path=), CompactMetadata.postTokens/durationMs, agentId on
                  BridgeStatusSystemRecord, apiErrorStatus on AssistantRecord, ApiError.type literal,
                  'auto' permissionMode (2.1.111), ExitWorktreeToolInput, MonitorToolInput (2.1.98+),
                  EnterWorktreeToolInput.path (2.1.105+), GlobToolInput.limit, Grep -o flag and
                  pattern/command AliasChoices, server_error error literal; expanded AttachmentData
                  union by 21 types (task_reminder, hook_success, hook_blocking_error,
                  hook_non_blocking_error, hook_additional_context, queued_command, dynamic_skill,
                  skill_listing, nested_memory, file, edited_text_file, opened_file_in_ide,
                  selected_lines_in_ide, diagnostics, plan_file_reference, plan_mode_exit,
                  compact_file_reference, command_permissions, date_change, auto_mode,
                  auto_mode_exit) (2.1.112+)
- Schema v0.2.23: Added NetworkError.type (always null in observed data; reserved/future field
                  mirroring ApiError.type), reverted v0.2.22's GrepToolInput pattern/command
                  AliasChoices (Opus 4.7 hallucination patched at source per new
                  'schema is the source of truth' rule in mcp/claude-session/CLAUDE.md §3),
                  extended CLAUDE_CODE_MAX_VERSION to 2.1.114 (2.1.113/114 verified clean)
- Schema v0.2.24: Filled in gaps observed against a larger corpus: SendMessagePromptToolInput
                  ({to, prompt} shape, 2.1.112+), PushNotificationToolInput ({message, status}
                  shape, 2.1.110+), PlanModeAttachment and PlanModeReentryAttachment (2.1.112+),
                  agentId on ScheduledTaskFireSystemRecord (sidechain subagents, 2.1.112+),
                  'fast' added to TokenUsage.speed Literal (2.1.112+), made EditToolInput.new_string
                  optional (aborted/partial tool calls)
- Schema v0.2.25: Closed the remaining tool-result fallback gaps so every Claude Code built-in
                  tool result parses into a typed model (true 100% validation, no MCPToolResult
                  fallthrough). Additions: BashToolResult.staleReadFileStateHint, isImage optional;
                  ReadPartsToolResult + ReadPartsFileInfo (large-PDF split), ReadFileUnchangedToolResult
                  + ReadFileUnchangedFileInfo (unchanged-since-last-read); EditToolResult.originalFile
                  optional (null when pre-edit contents not captured); WriteToolResult.userModified
                  optional; TaskToolResult.agentType + toolStats (with new TaskToolStats);
                  SkillToolResult.allowedTools optional; CronCreateToolResult (id, humanSchedule,
                  recurring, durable); ExitWorktreeToolResult (action Literal['keep','remove'],
                  originalCwd, worktreePath, worktreeBranch, message, plus discardedFiles/
                  discardedCommits when action='remove'). ExitWorktreeToolResult precedes
                  EnterWorktreeToolResult in the ToolResult union so the more-specific shape wins.
- Schema v0.2.26: Added InvokedSkillsAttachment for the 'invoked_skills' attachment type
                  surfacing user-invoked skills with their injected content (Claude Code 2.1.119,
                  stabilizing the skill-invocation path fixed by the 'skills invoked before
                  auto-compaction re-executed' changelog note). Widened McpMeta._meta from
                  EmptyDict to a typed McpMetaMetadata namespace containing FastMcpMeta
                  (wrap_result), after observing fastmcp-framework metadata surfacing on MCP
                  tool results from 2.1.118+ (likely coincident with hooks gaining the
                  `type: "mcp_tool"` invocation path). Extended CLAUDE_CODE_MAX_VERSION to
                  2.1.119.
- Schema v0.2.27: Added DeferredToolsDeltaAttachment for the 'deferred_tools_delta' attachment
                  type (Claude Code 2.1.120+) emitted by the tool-search/deferred-tools system
                  — shape mirrors McpInstructionsDeltaAttachment but with addedLines instead of
                  addedBlocks (a flat list of registered tool names, one per line). Made
                  LastPromptRecord.lastPrompt optional (str | None = None): 2.1.120+ also writes
                  a placeholder last-prompt record near session start where the prompt text is
                  absent until the first user input. Extended CLAUDE_CODE_MAX_VERSION to 2.1.121.
- Schema v0.2.28: Modeled message-level cache-miss diagnostics emitted on assistant messages
                  from Claude Code 2.1.119+ via the `cache-diagnosis-2026-04-07` Anthropic API
                  beta header: Message.diagnostics: MessageDiagnostics | None, with a
                  CacheMissReason discriminated union over three observed type variants —
                  CacheMissReasonPreviousMessageNotFound, CacheMissReasonSystemChanged (carries
                  cache_missed_input_tokens: int), and CacheMissReasonUnavailable. Added
                  AssistantRecord.attributionAgent (str | None) introduced 2.1.121+ for sub-agent
                  record attribution; typed str (matching agentName/teamName precedent) since
                  values are drawn from the user's agent registry rather than a Claude Code
                  internal enum. Added LastPromptRecord.leafUuid (str | None) — optional pointer
                  to the latest branch leaf UUID, parallel to SummaryRecord.leafUuid.
- Schema v0.2.29: Added AiTitleRecord for the `ai-title` record type written by Claude Code
                  2.1.122+ when generating session titles for the `/resume` UI (shape: type,
                  aiTitle, sessionId). Split the CacheMissReason discriminated union to cover
                  two new variants observed in the wild: CacheMissReasonMessagesChanged and
                  CacheMissReasonToolsChanged (both carry cache_missed_input_tokens: int) —
                  the prior CacheMissReasonSystemChanged is now narrowly the system-prompt
                  case. Added NestedMemoryContent.rawContent (str | None) capturing the
                  unprocessed CLAUDE.md text (front matter + body) alongside the parsed body.
                  Extended CLAUDE_CODE_MAX_VERSION to 2.1.123.
- Schema v0.2.30: Added AssistantRecord.attributionSkill (str | None) for assistant records
                  emitted while a skill is active; mirrors attributionAgent.
- Schema v0.2.31: Added HookInfo.durationMs (int) — hook execution duration in milliseconds,
                  always present in observed JSONL (100% prevalence) and confirmed in binary as
                  `durationMs:0` initializer; introduced ≤ 2.1.120. Single-field fix collapses
                  984 union-cascade errors on StopHookSummarySystemRecord (Pydantic left-to-right
                  union evaluation against a 16-field record). Added AssistantRecord.attributionPlugin
                  (str | None) paired with attributionSkill when a plugin-installed skill drives
                  the turn (e.g. "codex"); introduced 2.1.121 alongside attributionAgent /
                  attributionSkill. Added QueuedCommandAttachment.source_uuid (str | None) —
                  UUID of the source message that triggered a queued command; rare (1/355
                  globally), only on commandMode="prompt"; produced by the SDK replay path
                  (`SDKUserMessageReplay` events, Claude Code 2.1.20+ `replayUserMessages`).
                  Snake_case in wire schema, anomalous vs surrounding camelCase.
- Schema v0.2.32: Added EditedTextFileAttachment.displayPath (str | None) — display-relative
                  path alongside absolute filename, mirrors the displayPath field already modeled
                  on FileAttachment / SelectedLinesInIdeAttachment / CompactFileReferenceAttachment.
                  Modeled optional (~2% prevalence in observed JSONL, 2/107 records); confirmed in
                  binary (14 occurrences in 2.1.123 through 2.1.131 — pre-existing schema gap).
                  Added CacheMissReasonParamsChanged variant (type: 'params_changed', carries
                  cache_missed_input_tokens: int) to the CacheMissReason discriminated union;
                  server-emitted via the cache-diagnosis-2026-04-07 Anthropic API beta header,
                  not present in client binary (consistent with the other CacheMissReason
                  variants). Extended CLAUDE_CODE_MAX_VERSION to 2.1.131. Verified via parallel
                  changelog + binary diff workers that no new record types, system subtypes,
                  attachment types, message content types, or tool inputs were introduced
                  between 2.1.124 and 2.1.131; binary-only candidates (claude-sonnet-4-6-20251114
                  dated alias, service_tier='priority'/'batch' Zod widening, pause_turn
                  stop_reason) deferred per empirical-only modeling rule until observed in JSONL.
- Schema v0.2.33: Added AlreadyReadFileAttachment for the 'already_read_file' attachment type —
                  reminder injected when Claude re-Reads an unchanged file in the same session.
                  Shape mirrors FileAttachment exactly (filename, displayPath, FileAttachmentContent
                  wrapping FileAttachmentFileContent), so the nested types are reused directly.
                  Added CompactMetadata.preservedSegment (PreservedCompactSegment with
                  headUuid/anchorUuid/tailUuid) — anchor pointers identifying the conversation
                  segment preserved across an auto-compact; introduced 2.1.76. Single-field fix
                  collapsed 14 union-cascade errors on CompactBoundarySystemRecord (Pydantic
                  left-to-right union evaluation against a 16-field record). Corrected
                  preCompactDiscoveredTools annotation 2.1.81 → 2.1.76 from per-version binary
                  diff. Added DeferredToolsDeltaAttachment.readdedNames (tools removed then
                  re-registered) and pendingMcpServers (MCP servers still connecting at emit
                  time); both introduced in 2.1.128. Extended CLAUDE_CODE_MAX_VERSION to 2.1.138.
- Schema v0.2.34: Closed a cluster of long-standing schema gaps surfaced when a session
                  with team-mode records was cloned. (1) AgentListingDeltaAttachment for the
                  'agent_listing_delta' attachment type — agent registry delta carrying
                  addedTypes/removedTypes for identifiers, addedLines for formatted descriptions,
                  isInitial + showConcurrencyNote; shape mirrors DeferredToolsDeltaAttachment;
                  present in binary from at least 2.1.90. (2) teamName/agentName added to
                  AttachmentRecord — v0.2.15 added these to all record types but
                  AttachmentRecord (added in v0.2.20) was missed. (3) TaskReminderItem.owner
                  (str) and TaskReminderItem.metadata (Mapping[str, Any]) — binary schema:
                  `v.record(v.string(), v.unknown()).optional()`. Same metadata field added to
                  TaskCreateToolInput. (4) UserQuestion.multiSelect default False — binary
                  declares `v.boolean().default(!1)`, so the field is optional with default,
                  not required. (5) SendMessageToolInput rewritten to the post-backfill wire
                  shape (to, message, type, recipient, content, summary?) — Claude Code's
                  `backfillObservableInput` derives type/recipient/content from to/message
                  before JSONL serialization, so the recorded shape always carries all six
                  fields when type=='message'. Removed SendMessagePromptToolInput (v0.2.24)
                  per empirical-only modeling — zero observed records of {to, prompt} across
                  the 1108-session corpus, and no `prompt` field in the binary's SendMessage
                  zod schema. (6) TeamCreateToolInput.agent_type (str | None) — assigned to
                  team lead at creation (e.g., 'team-lead').
- Schema v0.2.35: Expanded ApiError.type and ApiErrorDetail.type literals to include
                  'rate_limit_error' alongside the existing 'overloaded_error' — both the
                  top-level ApiError.type (Claude Code 2.1.100+) and the nested
                  ApiErrorDetail.type. HTTP 429 rate-limit responses populate both fields with
                  'rate_limit_error', and the previous single-value Literal caused
                  ApiErrorSystemRecord validation to fall through the
                  ApiError | NetworkError | EmptyError union and land on BaseRecord (14
                  union-cascade extra_forbidden errors per failing record). Empirically:
                  108 outer + 108 inner 'rate_limit_error' occurrences across the local corpus;
                  binary string verification confirmed 'rate_limit_error' in Claude Code 2.1.138.
- Schema v0.2.36: Added HookCancelledAttachment for the 'hook_cancelled' attachment type emitted
                  when a hook is aborted before completion (observed in JSONL on Stop-hook
                  cancellation paths). Shape mirrors HookNonBlockingErrorAttachment minus the
                  stdout/stderr/exitCode trio that an unfinished hook can't produce: type,
                  hookName, toolUseID, hookEvent, command, durationMs (all required; 6/6 fields
                  present across observed records). Without this variant the AttachmentRecord
                  discriminated union rejects the record with union_tag_invalid, breaking
                  save_current_session / clone / archive for any session containing a cancelled
                  hook. Binary verification: 15 'hook_cancelled' occurrences in Claude Code 2.1.138,
                  16 in 2.1.77, 0 in 1.0.50 (introduced pre-2.1.77; exact version not pinned).
- Schema v0.2.37: Cluster fix surfaced when cloning a 2.1.156 session (clone crashed on the
                  unmodeled `mode` record) — closed every gap in the 2.1.139-2.1.156 window
                  (306 strict failures collapsed to 0 real schema gaps). New record/tool/attachment
                  types: SessionModeRecord (`mode` session-resumption sidecar, Claude Code 2.1.69+;
                  {type, mode, sessionId}; mode Literal['normal'], binary domain {normal, coordinator}
                  via isCoordinatorMode(); v0.2.20 modeled its permission-mode/worktree-state siblings
                  but missed this one); AgentsKilledSystemRecord (subtype=agents_killed, 2.1.139+, base
                  envelope only); WorkflowKeywordRequestAttachment (2.1.153+) and UltrathinkEffortAttachment
                  (2.1.139+), bare {type} markers; ScheduleWakeupToolInput (prompt/delaySeconds:int/reason).
                  New CacheMissReason variant model_changed (server diagnostic, mirrors existing variants).
                  New fields: claude-opus-4-8 in ModelId/_AllModelIds (2.1.154+); AssistantRecord.
                  attributionMcpServer/attributionMcpTool (2.1.149+); UserRecord.interruptedMessageId
                  (2.1.149+); TurnDurationSystemRecord.pendingBackgroundAgentCount (2.1.152+);
                  SkillListingAttachment.names; OpenedFileInIdeAttachment.displayPath; TokenUsage.
                  output_tokens_details (OutputTokensDetails.thinking_tokens, API passthrough);
                  EmptyError.type=None accepts the bare {"type": null} api_error shape. Every field
                  binary-verified or classified server-only-diagnostic per precedent; binary-only siblings
                  (pendingWorkflowCount, attributionSnapshots) and the 'coordinator' mode value deferred
                  per empirical-only modeling (no JSONL occurrences). Two ScheduleWakeup records carrying a
                  string `delaySeconds` (model coercion misfire vs the binary's zod number contract) were
                  data-patched, not accommodated. Added BridgeSessionRecord for the 'bridge-session'
                  sidecar ({type, sessionId, bridgeSessionId, lastSequenceNum}) emitted by
                  /remote-control bridge sessions — binary-confirmed, surfaced in JSONL once a live
                  bridge session ran (deferred at first pass per empirical-only: zero JSONL then).
                  Extended CLAUDE_CODE_MAX_VERSION to 2.1.156.
- Schema v0.2.38: Cross-machine gaps for Claude Code 2.1.157 — 8 modeled, 1 data-cleaned. forkedFrom:
                  ForkOrigin {sessionId, messageUuid} on BaseRecord — a session-fork backpointer
                  present on every record of a forked session (Claude Code 2.1.8+, binary-bisected).
                  MonitorToolInput timeout_ms/persistent
                  -> optional (binary: only description/command required). HookInfo.durationMs -> optional
                  (absent on pre-2.1.120 records; reverse of v0.2.31). ScheduledTaskFireSystemRecord +=
                  teamName/agentName (missed when v0.2.15 added them to all record types). SendMessage:
                  control bodies (SendMessageControl = shutdown_request | shutdown_response, discriminated)
                  widen SendMessageToolInput.message to str|control and type; SendMessageLegacyToolInput
                  for the 2.1.63 pre-refactor {type, recipient, content} shape. CompactMetadata.
                  preservedMessages (PreservedCompactMessages {anchorUuid, uuids, allUuids}).
                  ImageSource.media_type widened to the Anthropic 4-value enum (+image/gif, image/webp).
                  TurnDurationSystemRecord.pendingWorkflowCount — the binary-only sibling v0.2.37 deferred,
                  now observable once workflows ran. The lone misfire (a SendMessage {to, prompt}
                  hallucination, the shape v0.2.34 already removed) was data-cleaned, not modeled.
                  Extended CLAUDE_CODE_MAX_VERSION to 2.1.157.
- Schema v0.2.39: Workflow tool support + the session-nested workflow layout. SendMessageToolInput
                  content optional and request_id/approve added (the plan-approval / shutdown reply
                  fields the harness flattens onto the tool input). WorkflowToolInput (script?,
                  2.1.146+) and RemoteTriggerToolInput (action, 2.1.81+). WorkflowJournalRecord
                  (started | result, discriminated) + validate_journal_record for the run-journal
                  artifact (subagents/workflows/wf_<runId>/journal.jsonl), a distinct stream from the
                  session transcript. Agent recognition widened to Claude Code's affix-only rule
                  (agent-*.jsonl / .meta.json carry an opaque id; the typed-id type slug is cosmetic).
                  Operation support (archive format 2.4): the workflow-nested layout — workflow
                  agents, their .meta.json sidecars, run-journals, and the sibling <session>/workflows/
                  run metadata + scripts — now round-trips through clone/move/restore/archive/delete
                  with agentId / sessionId / scriptPath remap. session-memory tombstoned (removed
                  upstream ~2.1.128). Extended CLAUDE_CODE_MAX_VERSION to 2.1.159.
- Schema v0.2.40: Cross-machine tool/record gaps + Claude Code 2.1.160, all binary-confirmed.
                  BaseRecord.sessionKind (Literal['bg'] | None) — session origin kind from
                  CLAUDE_CODE_SESSION_KIND; interactive sessions omit it, only 'bg' observed in JSONL
                  (binary enum interactive|bg|daemon|daemon-worker). On BaseRecord, so it also resolves
                  a union-cascade across system records (its absence bounced them to BaseRecord).
                  WorktreeSessionData.worktreeBranch -> optional (absent when EnterWorktree attaches to
                  an existing worktree — no branch created). StructuredOutputToolInput for the Workflow
                  structured-output built-in (binary inputSchema z.object({}).passthrough(); re-typed
                  from the permissive fallback so it reads as a built-in rather than MCP).
                  WorkflowToolInput gained name/args/scriptPath (saved-workflow and on-disk-script
                  forms; args an arbitrary caller payload, typed pydantic.JsonValue). New
                  SendUserFileToolInput (deliver files to the user in remote environments, 2.1.142+).
                  TaskUpdateToolInput gained metadata. Genuine tool-input hallucinations were
                  data-cleaned, not modeled: SendMessage prompt->message; Grep string-typed
                  -n/-C/head_limit coerced to bool/int (the binary preprocesses strings but stores the
                  pre-coercion shape); Monitor stray run_in_background; an empty SendUserFile. The
                  2.1.159->2.1.160 release delta is record-surface-empty; watch item: DesignSync (new
                  first-party built-in tool, binary-confirmed, not yet in JSONL — needs a typed input
                  on first capture). Extended CLAUDE_CODE_MAX_VERSION to 2.1.160.
- Schema v0.2.41: Claude Code 2.1.162. UserRecord.promptSource (Literal queued|sdk|system|typed —
                  prompt origin, 2.1.161+). NetworkDownError + NetworkDownConnection: the network-down
                  api_error variant ({message, formatted, connection{code,message,isSSLError},
                  isNetworkDown, rateLimits}, 2.1.161), a fourth error-union member ahead of EmptyError.
                  queued_command attachment gained origin (UserRecordOrigin, 2.1.157); its commandMode
                  tightened to Literal plan|prompt|task-notification. WorkflowToolInput.resumeFromRunId.
                  UserRecordOrigin.kind tightened to the binary origin-producer set (auto-continuation|
                  channel|coordinator|peer|task-notification). Enum Literals now carry the full binary
                  producer set, not only JSONL-observed values — strict validation trips on genuine
                  post-binary drift, not on values the installed binary already emits. Extended
                  CLAUDE_CODE_MAX_VERSION to 2.1.162.
- Schema v0.2.42: Claude Code 2.1.163. Modeled the binary appendEntry record-type family the
                  empirical-only policy had skipped, all binary-proven with dispatch branches:
                  ForkContextRefRecord (subagent fork-context pointer; the stuck-session-delete
                  blocker), TagRecord, AgentColorRecord, AgentSettingRecord, IsolationLatchRecord,
                  SpeculationAcceptRecord, ContentReplacementRecord. Five sibling rT4 routing types
                  (frame-link, marble-origami-{commit,snapshot,reset}, attribution-snapshot) left
                  unmodeled — registered-but-unwired or dead code, no enumerable producer. Added
                  RetryableHttpError + RateLimitInfo: the retryable-HTTP api_error variant (429
                  rate_limit / 529 overloaded — status + requestId + the network-down envelope), a
                  fifth error-union member. StopHookSummarySystemRecord.hookAdditionalContext
                  (2.1.163). TaskStatusAttachment (background-execution status, distinct from the
                  TaskCreate TODO system). SessionModeRecord.mode + IsolationLatchRecord.side gained
                  'coordinator' (binary isCoordinatorMode projection; previously empirical-only-
                  deferred). Extended CLAUDE_CODE_MAX_VERSION to 2.1.163.
- Schema v0.2.43: NetworkDownError.connection -> nullable. The request-diagnostic api_error
                  envelope (formatted + isNetworkDown, no status) also carries request-timeouts
                  ({connection: null, isNetworkDown: false}); the binary's VU(H) connection builder
                  returns null when there is no socket-level failure. MAX unchanged (2.1.163).
- Schema v0.2.44: Claude Code 2.1.172 (Claude Fable 5 era). ModelRefusalFallbackSystemRecord
                  (subtype model_refusal_fallback: API-side refusal -> fallback retry; producer
                  ≤2.1.162, surfaced in JSONL by Fable refusal fallbacks 2.1.170+) and
                  ModelFallbackSystemRecord (subtype model_fallback: model_not_found |
                  permission_denied availability fallback; producer ≤2.1.162). ArtifactToolInput
                  (new built-in publishing .html/.md to a private claude.ai page; introduced
                  2.1.164-2.1.172). ModelId += claude-fable-5, claude-mythos-5 (2.1.170).
                  UserRecordOrigin.kind += human (2.1.172 adds a real producer on the
                  resume-background-agent path, retiring the consumer-guard-only note).
                  FallbackContent message block ({type:fallback, from{model}, to{model}};
                  server_fallback stream events are never persisted). UsageIteration gained
                  type fallback_message and a per-iteration model field.
                  frame-link routing key removed upstream (was never modeled).
                  AssistantRecord.error += model_not_found (404 model-unavailable API error;
                  producer present ≤2.1.172, a latent baseline gap surfaced empirically by a
                  2.1.173 session). Verified 2.1.173-2.1.181 schema-light: no new records, tools,
                  or model IDs; TeamCreate/TeamDelete tools removed upstream (2.1.178) but historical
                  records retained; DesignSync is an MCP-style claude.ai tool, not a CC built-in.
                  Extended CLAUDE_CODE_MAX_VERSION to 2.1.181.
- Schema v0.2.45: Claude Code 2.1.183. MAX-only bump — binary-proven no new session-schema
                  surface across 2.1.181-2.1.183 (identical built-in tool, model-ID, and
                  system-subtype sets; 2.1.182 is a notes-less point release, 2.1.183 is
                  behavioral/TUI/bugfix — the lone JSONL-adjacent item, scheduled-task/webhook
                  deliveries reclassed as task notifications, maps to the existing
                  UserRecordOrigin.kind=task-notification). Extended CLAUDE_CODE_MAX_VERSION to 2.1.183.
- If validation fails, Claude Code schema may have changed - update models accordingly

NEW FIELDS IN CLAUDE CODE 2.0.51+ (Schema v0.1.3):

slug field:
  Human-readable conversation identifier using adjective-verb-animal format
  (e.g., "jiggly-churning-rabbit", "floofy-singing-crab").
  - NOT a permanent session identifier - can change within same session file
  - Regenerated when session is resumed after extended inactivity (~hours)
  - Appears on: UserRecord, AssistantRecord, LocalCommandSystemRecord, ApiErrorSystemRecord
  - Some records have slug: null during transitions
  - Purpose: Makes sessions easier to reference than UUIDs
  - Related GitHub issues: #2112, #10943, #11408 (custom session naming requests)

context_management field:
  Tracks server-side context optimization operations on Message objects.
  Structure: {"applied_edits": []} - records which context clearing operations were executed.
  - Part of Claude API's context editing feature (Nov 2025)
  - Automatically clears older tool results when context exceeds thresholds
  - Helps maintain long-running sessions without context window exhaustion
  - Previously always null, now populated during context optimization

document content type:
  New MessageContent type for PDF uploads with structure:
  {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
  - Part of Claude API's PDF support, not Claude Code specific
  - Enables visual + textual understanding of PDF documents
  - Max 32MB, 100 pages per request

Skill tool:
  New tool for invoking Agent Skills - auto-discovered capabilities.
  Input: {"skill": "handoff", "args": "..."}
  - Skills differ from slash commands: auto-discovered vs explicit invocation
  - Skills are folders with SKILL.md + resources (templates, scripts, examples)
  - Claude examines available skills and uses them when relevant to task
  - Built-in skills: Excel, PowerPoint, Word, PDF form-filling

Key findings from analyzing real session files:
- Assistant records from agents/subprocesses have nested API response structure
- file-history-snapshot records don't inherit from BaseRecord (different schema)
- summary records don't have uuid/timestamp/sessionId (minimal schema)
- User records can have optional toolUseResult field with tool execution metadata
- Message structure varies: can be string or structured content list

Important pattern for unused/reserved fields:
- Fields that are present in JSON but ALWAYS null are typed as None (not Any | None)
- This includes: AssistantRecord.stopReason, Message.container
- These are reserved/future fields in the Claude API schema but currently unpopulated
- Using None type prevents accidental usage and makes the schema more accurate

Round-trip serialization:
- Use model_dump(exclude_unset=True, mode='json') to preserve original JSON structure
- This maintains null fields from input while excluding fields that were never set
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import pydantic
from cc_lib.schemas.base import EmptySequence
from cc_lib.types import CCVersion

from claude_session.schemas.session.markers import CCVersionStrField, PathField, PathListField
from claude_session.schemas.types import BaseStrictModel, ModelId, PermissiveModel

__all__ = [
    'CLAUDE_CODE_MAX_VERSION',
    'CLAUDE_CODE_MIN_VERSION',
    'SCHEMA_VERSION',
    'AgentColorRecord',
    'AgentListingDeltaAttachment',
    'AgentNameRecord',
    'AgentOutputToolInput',
    'AgentProgressData',
    'AgentSettingRecord',
    'AgentState',
    'AgentTeammateSpawnedResult',
    'AgentsKilledSystemRecord',
    'AgentsRetrievalResult',
    'AiTitleRecord',
    'AlreadyReadFileAttachment',
    'ApiError',
    'ApiErrorDetail',
    'ApiErrorResponse',
    'ApiErrorSystemRecord',
    'AppliedEdit',
    'ArtifactToolInput',
    'AskUserQuestionToolInput',
    'AskUserQuestionToolResult',
    'AssistantRecord',
    'AsyncTaskLaunchResult',
    'AttachmentData',
    'AttachmentRecord',
    'AutoModeAttachment',
    'AutoModeExitAttachment',
    'AwaySummarySystemRecord',
    'BackgroundTask',
    'BaseRecord',
    'BashOutputToolInput',
    'BashOutputToolResult',
    'BashProgressData',
    'BashToolInput',
    'BashToolResult',
    'BridgeSessionRecord',
    'BridgeStatusSystemRecord',
    'CacheCreation',
    'CacheMissReason',
    'CacheMissReasonMessagesChanged',
    'CacheMissReasonModelChanged',
    'CacheMissReasonParamsChanged',
    'CacheMissReasonPreviousMessageNotFound',
    'CacheMissReasonSystemChanged',
    'CacheMissReasonToolsChanged',
    'CacheMissReasonUnavailable',
    'ClearThinkingEdit',
    'CommandPermissionsAttachment',
    'CompactBoundarySystemRecord',
    'CompactFileReferenceAttachment',
    'CompactMetadata',
    'CompanionIntroAttachment',
    'ConnectionError',
    'ContentReplacement',
    'ContentReplacementRecord',
    'ContextManagement',
    'CronCreateToolInput',
    'CronCreateToolResult',
    'CronDeleteToolInput',
    'CronListToolInput',
    'CustomTitleRecord',
    'DateChangeAttachment',
    'DeferredToolsDeltaAttachment',
    'DiagnosticFile',
    'DiagnosticItem',
    'DiagnosticPosition',
    'DiagnosticSeverityRange',
    'DiagnosticsAttachment',
    'DocumentContent',
    'DocumentSource',
    'DynamicSkillAttachment',
    'EditToolInput',
    'EditToolResult',
    'EditedTextFileAttachment',
    'EmptyError',
    'EnterPlanModeToolInput',
    'EnterPlanModeToolResult',
    'EnterWorktreeToolInput',
    'EnterWorktreeToolResult',
    'ExitPlanModeToolInput',
    'ExitPlanModeToolResult',
    'ExitWorktreeToolInput',
    'ExitWorktreeToolResult',
    'FallbackContent',
    'FallbackModelRef',
    'FastMcpMeta',
    'FileAttachment',
    'FileAttachmentContent',
    'FileAttachmentFileContent',
    'FileBackupInfo',
    'FileHistorySnapshot',
    'FileHistorySnapshotRecord',
    'FileInfo',
    'ForkContextRefRecord',
    'ForkOrigin',
    'GlobToolInput',
    'GlobToolResult',
    'GrepToolInput',
    'GrepToolResult',
    'HandoffCommandResult',
    'HookAdditionalContextAttachment',
    'HookBlockingErrorAttachment',
    'HookBlockingErrorData',
    'HookCancelledAttachment',
    'HookInfo',
    'HookNonBlockingErrorAttachment',
    'HookProgressData',
    'HookSuccessAttachment',
    'ImageContent',
    'ImageDimensions',
    'ImageFileInfo',
    'ImageSource',
    'InformationalSystemRecord',
    'InvokedSkill',
    'InvokedSkillsAttachment',
    'IsolationLatchRecord',
    'KillShellMessageResult',
    'KillShellToolInput',
    'KillShellToolResult',
    'LSPOperation',
    'LSPToolInput',
    'LSPToolResult',
    'LastPromptRecord',
    'ListMcpResourcesToolInput',
    'LocalCommandSystemRecord',
    'MCPSearchToolInput',
    'MCPStructuredContent',
    'MCPToolInput',
    'MCPToolResult',
    'McpInstructionsDeltaAttachment',
    'McpMeta',
    'McpMetaMetadata',
    'McpProgressCompletedData',
    'McpProgressFailedData',
    'McpProgressStartedData',
    'McpResource',
    'McpResourceContent',
    'Message',
    'MessageContent',
    'MessageDiagnostics',
    'MicrocompactBoundarySystemRecord',
    'MicrocompactMetadata',
    'ModelFallbackSystemRecord',
    'ModelRefusalFallbackSystemRecord',
    'MonitorToolInput',
    'NestedMemoryAttachment',
    'NestedMemoryContent',
    'NetworkDownConnection',
    'NetworkDownError',
    'NetworkError',
    'NotebookEditToolInput',
    'OpenedFileInIdeAttachment',
    'OutputTokensDetails',
    'PatchHunk',
    'PdfFileInfo',
    'PermissionModeRecord',
    'PlanFileReferenceAttachment',
    'PlanModeAttachment',
    'PlanModeExitAttachment',
    'PlanModeReentryAttachment',
    'PrLinkRecord',
    'PreservedCompactMessages',
    'PreservedCompactSegment',
    'ProgressData',
    'ProgressRecord',
    'PromptPermission',
    'PushNotificationToolInput',
    'QueryUpdateData',
    'QuestionAnnotation',
    'QuestionOption',
    'QueueOperationRecord',
    'QueuedCommandAttachment',
    'RateLimitInfo',
    'ReadFileUnchangedFileInfo',
    'ReadFileUnchangedToolResult',
    'ReadImageToolResult',
    'ReadMcpResourceToolInput',
    'ReadMcpResourceToolResult',
    'ReadPartsFileInfo',
    'ReadPartsToolResult',
    'ReadPdfToolResult',
    'ReadTextToolResult',
    'ReadToolInput',
    'RemoteTriggerToolInput',
    'RetryableHttpError',
    'SavedHookContextRecord',
    'ScheduleWakeupToolInput',
    'ScheduledTaskFireSystemRecord',
    'SearchResultsReceivedData',
    'SelectedLinesInIdeAttachment',
    'SendMessageControl',
    'SendMessageLegacyToolInput',
    'SendMessageRouting',
    'SendMessageShutdownRequest',
    'SendMessageShutdownResponse',
    'SendMessageSimpleToolInput',
    'SendMessageToolInput',
    'SendMessageToolResult',
    'SendUserFileToolInput',
    'ServerToolUse',
    'SessionAnalysis',
    'SessionMetadata',
    'SessionModeRecord',
    'SessionRecord',
    'SessionRecordAdapter',
    'SimpleThinkingMetadata',
    'SkillListingAttachment',
    'SkillToolInput',
    'SkillToolResult',
    'SpeculationAcceptRecord',
    'StatusChange',
    'StopHookSummarySystemRecord',
    'StrictModel',
    'StructuredOutputToolInput',
    'SummaryRecord',
    'SystemRecord',
    'SystemSubtypeRecord',
    'TagRecord',
    'Task',
    'TaskCreateToolInput',
    'TaskGetItem',
    'TaskGetToolResult',
    'TaskListItem',
    'TaskListToolInput',
    'TaskListToolResult',
    'TaskOutputPollingResult',
    'TaskOutputToolInput',
    'TaskReminderAttachment',
    'TaskReminderItem',
    'TaskSingleItem',
    'TaskSingleToolResult',
    'TaskStatusAttachment',
    'TaskStopToolResult',
    'TaskToolInput',
    'TaskToolResult',
    'TaskToolStats',
    'TaskUpdateSuccessResult',
    'TaskUpdateToolInput',
    'TeamCreateToolInput',
    'TeamCreateToolResult',
    'TextContent',
    'ThinkingContent',
    'ThinkingMetadata',
    'ThinkingTrigger',
    'TodoItem',
    'TodoToolResult',
    'TodoWriteToolInput',
    'TokenUsage',
    'ToolInput',
    'ToolReferenceContent',
    'ToolResult',
    'ToolResultContent',
    'ToolResultContentBlock',
    'ToolSearchToolResult',
    'ToolUseCaller',
    'ToolUseContent',
    'TurnDurationSystemRecord',
    'UltrathinkEffortAttachment',
    'UsageIteration',
    'UserQuestion',
    'UserRecord',
    'UserRecordOrigin',
    'WaitingForTaskData',
    'WebFetchToolInput',
    'WebFetchToolResult',
    'WebSearchNestedResult',
    'WebSearchResult',
    'WebSearchResultWrapper',
    'WebSearchToolInput',
    'WebSearchToolResult',
    'WorkflowJournalRecord',
    'WorkflowJournalResult',
    'WorkflowJournalStarted',
    'WorkflowKeywordRequestAttachment',
    'WorkflowToolInput',
    'WorktreeSessionData',
    'WorktreeStateRecord',
    'WriteToolInput',
    'WriteToolResult',
    'validate_journal_record',
    'validate_session_record',
    'validated_copy',
]

# -- Schema Version ------------------------------------------------------------

SCHEMA_VERSION = '0.2.45'
CLAUDE_CODE_MIN_VERSION = CCVersion('2.0.35')
CLAUDE_CODE_MAX_VERSION = CCVersion('2.1.183')


# -- Base Configuration --------------------------------------------------------


class StrictModel(BaseStrictModel):
    """Session-layer strict model.

    Inherits from BaseStrictModel (extra='forbid', strict=True, frozen=True).
    Domain-specific customization can be added here if needed.
    """


# -- Message Content Types (Discriminated Union) -------------------------------


class ThinkingContent(StrictModel):
    """Thinking content block from assistant messages."""

    type: Literal['thinking']
    thinking: str
    signature: str  # Always non-null in observed data (31,105/31,105)


class TextContent(StrictModel):
    """Text content block from user or assistant messages."""

    type: Literal['text']
    text: str


# -- Tool Use Input Types (for tools that use file paths) ----------------------


class ReadToolInput(StrictModel):
    """Input for Read tool.

    Fields:
        file_path: Absolute path to the file to read
        limit: Maximum number of lines to read (for large files)
        offset: Line number to start reading from (1-indexed)
    """

    file_path: PathField
    limit: int | str | None = None  # Max lines to read
    offset: int | str | None = None  # Start line (1-indexed), can be malformed string like "\\248"
    pages: str | None = None  # Page range for PDF files (e.g. '1-10') (Claude Code 2.1.41+)


class WriteToolInput(StrictModel):
    """Input for Write tool."""

    file_path: PathField
    content: str


class EditToolInput(StrictModel):
    """Input for Edit tool."""

    file_path: PathField
    old_string: str
    new_string: str | None = None  # Absent on aborted/partial tool calls (Claude Code 2.1.112+)
    replace_all: bool | Literal['false'] | None = None


class SkillToolInput(StrictModel):
    """Input for Skill tool.

    Fields:
        skill: Skill name to invoke (e.g., 'handoff', 'commit')
        args: Optional arguments for the skill
    """

    skill: str
    args: str | None = None


class EnterWorktreeToolInput(StrictModel):
    """Input for EnterWorktree tool - creates an isolated git worktree."""

    name: str | None = None  # Optional worktree name
    path: str | None = None  # Path to existing worktree to attach to (Claude Code 2.1.105+)


class ExitWorktreeToolInput(StrictModel):
    """Input for ExitWorktree tool - exits the active worktree session (Claude Code 2.1.x)."""

    action: Literal['keep', 'remove']  # keep retains the worktree; remove deletes it
    discard_changes: bool | None = None  # When true with remove, discard uncommitted changes


class MonitorToolInput(StrictModel):
    """Input for Monitor tool - streams events from a long-running background script (Claude Code 2.1.98+)."""

    description: str  # Short human-readable description of what is being monitored
    command: str  # Shell command or script to run
    timeout_ms: int | None = None  # Kill deadline in ms; omitted when persistent=True (binary: optional, min 1000)
    persistent: bool | None = (
        None  # Runs for the session lifetime, ignoring timeout_ms (binary: optional, default false)
    )


class PushNotificationToolInput(StrictModel):
    """Input for PushNotification tool - sends a desktop/mobile notification (Claude Code 2.1.110+)."""

    message: str  # Notification body (under 200 chars for mobile)
    status: Literal['proactive']  # Only 'proactive' observed


class EnterPlanModeToolInput(StrictModel):
    """Input for EnterPlanMode tool (no parameters)."""


class AgentOutputToolInput(StrictModel):
    """Input for AgentOutputTool - retrieves output from background agents."""

    agentId: str  # Agent ID to retrieve output from
    block: bool | None = None  # Whether to block waiting for agent completion
    wait_up_to: int | None = None  # Optional timeout in seconds


class TaskOutputToolInput(StrictModel):
    """Input for TaskOutput tool - retrieves output from running/completed tasks."""

    task_id: str  # Task ID to retrieve output from
    block: bool | None = None  # Whether to block waiting for task completion
    timeout: int | None = None  # Optional timeout in milliseconds


# -- Bash Tool Input (23,236x occurrences) -------------------------------------


class BashToolInput(StrictModel):
    """Input for Bash tool - executes shell commands.

    Fields:
        command: The shell command to execute (required)
        description: Human-readable description in 5-10 words (usually present, but optional)
        timeout: Timeout in milliseconds (max 600000, default 120000)
        run_in_background: Run command asynchronously
        dangerouslyDisableSandbox: Bypass sandbox protection (requires policy permission)
    """

    command: str
    description: str | None = None  # Usually present (1078/1080) but some old records lack it
    timeout: int | str | None = None
    run_in_background: bool | None = None
    dangerouslyDisableSandbox: bool | None = None


# -- Grep Tool Input (2,909x occurrences) --------------------------------------


class GrepToolInput(StrictModel):
    """Input for Grep tool - searches file contents using ripgrep.

    Fields:
        pattern: Regex pattern to search for (required)
        path: Directory or file to search in (defaults to cwd)
        output_mode: content | files_with_matches | count
        glob: Glob pattern to filter files (e.g., "*.py")
        type: File type filter (e.g., "py", "js")
        multiline: Enable multiline regex mode
        head_limit: Limit output to first N results
        offset: Skip first N results
        context_lines: Number of context lines (alternative to -C)
        flags: String-format flags (e.g., "-i")
        grep: Alternative flag format (e.g., "-n")
        Hyphenated flags: -n (line numbers), -A/-B/-C (context), -i (case insensitive)
    """

    pattern: str
    path: PathField | None = None
    output_mode: Literal['content', 'files_with_matches', 'count', 'context'] | None = None
    glob: str | None = None
    type: str | None = None  # noqa: A003 - matches ripgrep's --type flag
    multiline: bool | None = None
    head_limit: int | None = None
    offset: int | None = None
    context: int | str | None = None  # Context lines - can be int or flag reference string like "-A"
    context_lines: int | str | None = None  # Explicit context lines count (alternative to -C)
    flags: str | None = None  # String-format flags (e.g., "-i")
    grep: str | None = None  # Alternative flag format (e.g., "-n") - legacy/variant usage
    # Hyphenated ripgrep flags (use Field alias for JSON compatibility)
    dash_n: bool | None = pydantic.Field(None, alias='-n')
    dash_A: int | None = pydantic.Field(None, alias='-A')
    dash_B: int | None = pydantic.Field(None, alias='-B')
    dash_C: int | None = pydantic.Field(None, alias='-C')
    dash_i: bool | None = pydantic.Field(None, alias='-i')
    dash_r: bool | None = pydantic.Field(None, alias='-r')
    dash_o: bool | None = pydantic.Field(None, alias='-o')  # --only-matching (ripgrep)


# -- Glob Tool Input (2,507x occurrences) --------------------------------------


class GlobToolInput(StrictModel):
    """Input for Glob tool - file pattern matching.

    Fields:
        pattern: Glob pattern to match files (e.g., "**/*.py")
        path: Directory to search in (defaults to cwd)
        limit: Max number of files to return (Claude Code 2.1.100+)
    """

    pattern: str
    path: PathField | None = None
    limit: int | None = None


# -- Task Tool Input (872x occurrences) ----------------------------------------


class TaskToolInput(StrictModel):
    """Input for Task tool - launches subagent for autonomous tasks.

    Fields:
        prompt: Task instructions for the subagent (required)
        subagent_type: Agent type - "Explore", "general-purpose", etc. (required)
        description: Human-readable task description for UI (usually present but optional)
        allowed_tools: Tools to grant the subagent (e.g., ["Write", "Bash"])
        run_in_background: Run asynchronously, check with TaskOutput
        model: Override model (e.g., "haiku", "sonnet")
        resume: Agent ID to resume from previous execution
        mode: Permission mode for the agent (Claude Code 2.1.19+)
    """

    description: str | None = None  # Usually present but some early records lack it
    prompt: str
    subagent_type: str | None = None  # Agent type - may be absent in newer versions (2.1.80+)
    allowed_tools: Sequence[str] | None = None  # Tools to grant the subagent
    run_in_background: bool | None = None
    model: str | None = None
    resume: str | None = None
    mode: Literal['default', 'bypassPermissions'] | None = None  # Permission mode (2.1.19+)
    max_turns: int | None = None  # Maximum agentic turns before stopping (2.1.25+)
    name: str | None = None  # Agent name within team (team mode only)
    team_name: str | None = None  # Team name (team mode only)
    isolation: str | None = None  # Isolation mode (e.g., "worktree") for Agent tool


# -- TeamCreate Tool Input (Claude Code 2.1.63+) -------------------------------


class TeamCreateToolInput(StrictModel):
    """Input for TeamCreate tool - creates a multi-agent team."""

    team_name: str
    description: str
    agent_type: str | None = None  # Agent type assigned to team lead (e.g., 'team-lead')


# -- SendMessage Tool Input (Claude Code 2.1.63+) ------------------------------


class SendMessageShutdownRequest(StrictModel):
    """Control-message body requesting a teammate shut down."""

    type: Literal['shutdown_request']
    reason: str | None = None


class SendMessageShutdownResponse(StrictModel):
    """Control-message body answering a shutdown request."""

    type: Literal['shutdown_response']
    request_id: str
    approve: bool
    reason: str | None = None


SendMessageControl = Annotated[
    SendMessageShutdownRequest | SendMessageShutdownResponse,
    pydantic.Field(discriminator='type'),
]


class SendMessageToolInput(StrictModel):
    """Input for SendMessage tool — post-backfill wire shape.

    Claude Code's `backfillObservableInput` derives top-level fields from `to`/`message`.
    For an object `message` (control body): `type` = message.type, `recipient` = to (always),
    `request_id`/`approve` are copied up iff present, and `content` = `message.reason ?? message.feedback`
    (so `content` is absent when the control body carries neither). `reason`/`feedback` never appear as
    top-level keys. For a string `message`: `type='message'`, `recipient` = to, `content` = message.
    The raw two-field form is SendMessageSimpleToolInput.
    """

    to: str
    message: str | SendMessageControl
    type: Literal['message', 'shutdown_request', 'shutdown_response']  # Mirrors message.type
    recipient: str  # Backfilled from to (always)
    content: str | None = None  # Backfilled: string message, or control reason/feedback; absent otherwise
    summary: str | None = None  # Passthrough inputSchema field: v.string().optional()
    request_id: str | None = None  # Backfilled from message.request_id (control _response bodies)
    approve: bool | None = None  # Backfilled from message.approve (control _response bodies)


class SendMessageSimpleToolInput(StrictModel):
    """Simplified SendMessage input format (Claude Code 2.1.81+)."""

    to: str
    message: str


class SendMessageLegacyToolInput(StrictModel):
    """Pre-refactor SendMessage wire shape (Claude Code 2.1.63): type/recipient/content, no to/message."""

    type: Literal['message']
    recipient: str
    content: str
    summary: str | None = None


# -- TaskCreate Tool Input (Claude Code 2.1.17+) -------------------------------


class TaskCreateToolInput(StrictModel):
    """Input for TaskCreate tool - creates a new task in the task list.

    Fields:
        subject: Brief title for the task (imperative form, e.g., "Run tests")
        description: Detailed description of what needs to be done
        activeForm: Present continuous form shown in spinner (e.g., "Running tests")
        metadata: Arbitrary caller metadata; binary schema: v.record(v.string(), v.unknown()).optional()
    """

    subject: str
    description: str
    activeForm: str | None = None
    metadata: Mapping[str, Any] | None = (
        None  # strict_typing_linter.py: loose-typing — binary schema is v.record(v.string(), v.unknown()).optional()
    )


# -- TaskUpdate Tool Input (Claude Code 2.1.17+) -------------------------------


class TaskUpdateToolInput(StrictModel):
    """Input for TaskUpdate tool - updates an existing task.

    Fields:
        taskId: ID of the task to update (required)
        status: New status (pending, in_progress, completed)
        description: New description for the task
        owner: Agent/worker name to assign the task to
        addBlockedBy: Task IDs that block this task
        metadata: Arbitrary caller metadata; binary schema: v.record(v.string(), v.unknown()).optional()
    """

    taskId: str
    status: Literal['pending', 'in_progress', 'completed', 'deleted'] | None = None
    subject: str | None = None  # Updated task title
    description: str | None = None  # Updated task description
    activeForm: str | None = None  # Updated spinner text
    owner: str | None = None
    addBlocks: Sequence[str] | None = None  # Task IDs that this task blocks
    addBlockedBy: Sequence[str] | None = None
    metadata: Mapping[str, Any] | None = (
        None  # strict_typing_linter.py: loose-typing — binary schema is v.record(v.string(), v.unknown()).optional()
    )


# -- TaskList Tool Input (Claude Code 2.1.17+) ---------------------------------


class TaskListToolInput(StrictModel):
    """Input for TaskList tool - lists all tasks (no parameters)."""


# -- TodoWrite Tool Input (3,186x occurrences) ---------------------------------


class TodoWriteToolInput(StrictModel):
    """Input for TodoWrite tool - tracks task progress.

    Fields:
        todos: List of todo items, each with content/status/activeForm
    """

    todos: Sequence[TodoItem]


# -- WebSearch Tool Input (284x occurrences) -----------------------------------


class WebSearchToolInput(StrictModel):
    """Input for WebSearch tool - web search.

    Fields:
        query: Search query string (required)
        allowed_domains: Only include results from these domains
        blocked_domains: Exclude results from these domains
    """

    query: str
    allowed_domains: Sequence[str] | None = None
    blocked_domains: Sequence[str] | None = None


# -- WebFetch Tool Input (177x occurrences) ------------------------------------


class WebFetchToolInput(StrictModel):
    """Input for WebFetch tool - fetches URL content.

    Fields:
        url: URL to fetch (required)
        prompt: Prompt to run on fetched content (required)
    """

    url: str
    prompt: str


# -- BashOutput Tool Input (192x occurrences) ----------------------------------


class BashOutputToolInput(StrictModel):
    """Input for BashOutput tool - retrieves background bash output.

    Fields:
        bash_id: Background task ID (required)
        block: Whether to wait for completion
        filter: Regex pattern to filter output
        wait_up_to: Max seconds to wait (for blocking mode)
    """

    bash_id: str
    block: bool | None = None
    filter: str | None = None  # noqa: A003 - matches tool's field name
    wait_up_to: int | None = None


# -- AskUserQuestion Tool Input (159x occurrences) -----------------------------


class AskUserQuestionToolInput(StrictModel):
    """Input for AskUserQuestion tool - asks user multiple choice questions.

    Fields:
        questions: List of questions (1-4), each with question/header/options/multiSelect
        answers: User answers (populated by permission component)
    """

    questions: Sequence[UserQuestion]
    answers: Mapping[str, str] | None = None


# -- ExitPlanMode Tool Input (150x occurrences) --------------------------------


class PromptPermission(StrictModel):
    """A prompt-based permission request in ExitPlanMode."""

    tool: str
    prompt: str


class ExitPlanModeToolInput(StrictModel):
    """Input for ExitPlanMode tool - exits planning mode.

    Note: This tool can be invoked with empty input {} or with plan/launchSwarm.
    The empty variant signals plan approval request.

    Fields:
        plan: Plan content in markdown (optional)
        launchSwarm: Whether to launch swarm agents (optional)
        allowedPrompts: Prompt-based permission requests (tool + prompt pairs)
        pushToRemote: Whether to push plan to remote Claude.ai session
    """

    plan: str | None = None
    launchSwarm: bool | None = None
    allowedPrompts: Sequence[PromptPermission] | None = None
    pushToRemote: bool | None = None
    planFilePath: str | None = None


# -- KillShell Tool Input (58x occurrences) ------------------------------------


class KillShellToolInput(StrictModel):
    """Input for KillShell tool - terminates a running shell.

    Fields:
        shell_id: ID of the shell to kill (required)
    """

    shell_id: str


# -- Cron Tool Inputs (Claude Code 2.1.71+) ------------------------------------


class CronCreateToolInput(StrictModel):
    """Input for CronCreate tool - schedules recurring prompts."""

    cron: str
    prompt: str
    recurring: bool | None = None
    durable: bool | None = None


class CronDeleteToolInput(StrictModel):
    """Input for CronDelete tool - cancels a scheduled cron job."""

    id: str


class CronListToolInput(StrictModel):
    """Input for CronList tool - lists scheduled cron jobs."""


class ScheduleWakeupToolInput(StrictModel):
    """Input for ScheduleWakeup tool - arms a /loop dynamic-mode wake heartbeat.

    delaySeconds is the API contract type (zod number, clamped to [60, 3600]
    by the runtime).
    """

    prompt: str
    delaySeconds: int
    reason: str


# -- Workflow + RemoteTrigger Tool Inputs --------------------------------------


class WorkflowToolInput(StrictModel):
    """Input for the Workflow tool — multi-agent orchestration (Claude Code 2.1.146+).

    Binary inputSchema: {script?, name?, scriptPath?, resumeFromRunId?, args?,
    description?, title?} — all optional (a refine requires one of script/name/scriptPath).
    The inline-script (`script`), on-disk-script (`scriptPath`), and saved-workflow (`name` + `args`)
    forms all appear in observed data; description/title remain binary-only and unmodeled.
    `args` is an arbitrary caller-supplied JSON value passed verbatim to the run.
    """

    script: str | None = None
    name: str | None = None  # Saved-workflow name (the name+args form runs a registered workflow)
    args: pydantic.JsonValue | None = None  # Arbitrary caller-supplied input exposed to the workflow as `args`
    scriptPath: str | None = None  # Path to a persisted workflow script on disk; takes precedence over script/name
    resumeFromRunId: str | None = None  # Resume a prior workflow run by id; completed agents replay cached results


class RemoteTriggerToolInput(StrictModel):
    """Input for the RemoteTrigger tool — claude.ai routines API (Claude Code 2.1.81+).

    Binary inputSchema: {action: enum[list|get|create|update|run] (required),
    trigger_id?, body?}. Only `action` appears in observed session data
    (`{"action": "list"}`), so it alone is modeled.
    """

    action: Literal['list', 'get', 'create', 'update', 'run']


class SendUserFileToolInput(StrictModel):
    """Input for the SendUserFile tool — delivers files to the user (Claude Code 2.1.142+).

    Binary inputSchema: {files: array(string).min(1) (required), status: enum[normal|proactive]
    (required), caption?}. Surfaces artifacts (screenshots, reports) — `proactive` pushes
    unprompted, `normal` accompanies a reply.
    """

    files: Sequence[str]  # File paths (absolute or relative to cwd); binary requires >= 1
    status: Literal['normal', 'proactive']
    caption: str | None = None  # Optional short caption for the file(s)


# -- ListMcpResourcesTool Input (5x occurrences) -------------------------------


class ListMcpResourcesToolInput(StrictModel):
    """Input for ListMcpResourcesTool - lists available MCP resources.

    Fields:
        server: Filter by specific MCP server name (optional)
    """

    server: str | None = None


# -- NotebookEdit Tool Input ---------------------------------------------------


class NotebookEditToolInput(StrictModel):
    """Input for NotebookEdit tool - edits Jupyter notebook cells.

    Fields:
        notebook_path: Absolute path to the .ipynb file (required)
        new_source: New source content for the cell (required)
        cell_id: ID of cell to edit (optional)
        cell_type: Type of cell - code or markdown
        edit_mode: replace | insert | delete
    """

    notebook_path: PathField
    new_source: str
    cell_id: str | None = None
    cell_type: Literal['code', 'markdown'] | None = None
    edit_mode: Literal['replace', 'insert', 'delete'] | None = None


# -- ReadMcpResource Tool Input ------------------------------------------------


class ReadMcpResourceToolInput(StrictModel):
    """Input for ReadMcpResourceTool - reads a specific MCP resource.

    Fields:
        server: MCP server name (required)
        uri: Resource URI to read (required)
    """

    server: str
    uri: str


# -- LSP Tool Input (Language Server Protocol operations) ----------------------


LSPOperation = Literal[
    'goToDefinition',
    'findReferences',
    'hover',
    'documentSymbol',
    'workspaceSymbol',
    'goToImplementation',
    'prepareCallHierarchy',
    'incomingCalls',
    'outgoingCalls',
]


class LSPToolInput(StrictModel):
    """Input for LSP tool - Language Server Protocol operations.

    Fields:
        operation: LSP operation to perform
        filePath: Path to the file to operate on
        line: Line number (1-based as shown in editors)
        character: Character offset (1-based as shown in editors)
    """

    operation: LSPOperation
    filePath: PathField
    line: int
    character: int


# -- MCPSearch Tool Input ------------------------------------------------------


class MCPSearchToolInput(StrictModel):
    """Input for MCPSearch tool - searches for MCP tools.

    Fields:
        query: Search query string, can use 'select:' prefix for exact tool selection
        max_results: Maximum number of results to return (optional, can be int or string)
    """

    query: str
    max_results: int | str | None = None  # Can be "1" string or 1 int depending on serialization


# -- MCP Tool Input (Third-Party Tools) ----------------------------------------


class MCPToolInput(PermissiveModel):
    """Permissive model for MCP (Model Context Protocol) tool inputs.

    MCP tools are third-party integrations with varying schemas. We intentionally
    do not model their specific fields because:
    1. There are 64+ different MCP tools with different schemas
    2. They are user-configured and can change without Claude Code updates
    3. Their schemas are defined by external MCP servers

    The validator on ToolUseContent enforces that ONLY tools with names starting
    with 'mcp__' can use this model. Claude Code built-in tools MUST have typed
    models - if one falls through to MCPToolInput, the validator raises an error.

    Uses PermissiveModel to accept any fields while maintaining type observability.
    """


class ArtifactToolInput(StrictModel):
    """Input for Artifact tool - publishes an .html/.md file to a private claude.ai page.

    Binary strictObject ``{file_path, favicon (one or two emoji), label?, url?}``.
    Introduced between Claude Code 2.1.163 and 2.1.172 (absent in 2.1.163).
    """

    file_path: PathField
    favicon: str
    label: str | None = None
    url: str | None = None


class StructuredOutputToolInput(PermissiveModel):
    """Input for the StructuredOutput built-in (Workflow structured output, 2.1.146+).

    The tool's binary inputSchema is z.object({}).passthrough() — an arbitrary, caller-defined
    payload (the workflow's per-call JSON Schema), so no fields are modeled. A distinct named type
    (vs the MCP fallback) keeps the artifact observably a Claude Code built-in; ToolUseContent
    re-types it from the permissive fallback — it is never matched directly, since MCPToolInput
    catches first in the left-to-right union.
    """


# Union of tool inputs (typed models first, PermissiveModel fallback for MCP tools)
# NOTE: Order matters! More specific (more required fields) should come first.
# Models with no required fields must come last before fallback.
ToolInput = Annotated[
    # Path-based tools (most specific - file_path + other required fields)
    WriteToolInput  # file_path, content required
    | EditToolInput  # file_path, old_string, new_string required
    | NotebookEditToolInput  # notebook_path, new_source required
    | ReadToolInput  # file_path required
    | ArtifactToolInput  # file_path, favicon required
    # Multi-field tools
    | SendMessageToolInput  # to, message, type, recipient, content required (backfilled wire shape)
    | SendMessageSimpleToolInput  # to, message required (2.1.81+)
    | SendMessageLegacyToolInput  # type, recipient, content required; no to/message (2.1.63 pre-refactor)
    | SendUserFileToolInput  # files, status required (2.1.142+)
    | TaskToolInput  # prompt, description, subagent_type required
    | TaskCreateToolInput  # subject, description required (2.1.17+)
    | TeamCreateToolInput  # team_name, description required (2.1.63+)
    | BashToolInput  # command required (description optional since some old records lack it)
    | GrepToolInput  # pattern required, has custom model_config
    | GlobToolInput  # pattern required
    | WebFetchToolInput  # url, prompt required
    | ReadMcpResourceToolInput  # server, uri required
    | AskUserQuestionToolInput  # questions required
    | TodoWriteToolInput  # todos required
    | WebSearchToolInput  # query required
    | MCPSearchToolInput  # query required (new in 2.1.4)
    | LSPToolInput  # operation, filePath, line, character required (ENABLE_LSP_TOOL=1)
    # Single-field tools
    | AgentOutputToolInput  # agentId required
    | TaskOutputToolInput  # task_id required
    | TaskUpdateToolInput  # taskId required (2.1.17+)
    | BashOutputToolInput  # bash_id required
    | KillShellToolInput  # shell_id required
    | CronCreateToolInput  # cron, prompt required (2.1.71+)
    | CronDeleteToolInput  # id required (2.1.71+)
    | ScheduleWakeupToolInput  # prompt, delaySeconds, reason required
    | SkillToolInput  # skill required
    | RemoteTriggerToolInput  # action required (2.1.81+)
    # Optional-only fields (must be near end)
    | ExitPlanModeToolInput  # plan optional, launchSwarm optional
    | ListMcpResourcesToolInput  # server optional
    | EnterWorktreeToolInput  # name optional (2.1.63+)
    | ExitWorktreeToolInput  # action required (2.1.105+)
    | MonitorToolInput  # description, timeout_ms, persistent, command required (2.1.98+)
    | PushNotificationToolInput  # message, status required (2.1.110+)
    | WorkflowToolInput  # script optional, no required fields (2.1.146+)
    | TaskListToolInput  # No fields (2.1.17+)
    | CronListToolInput  # No fields (2.1.71+)
    | EnterPlanModeToolInput  # No fields - must be last before fallback!
    | MCPToolInput  # Fallback for MCP tools (PermissiveModel for observability)
    | StructuredOutputToolInput,  # Open-schema built-in; never matched directly (MCP catches first), re-typed by ToolUseContent
    pydantic.Field(union_mode='left_to_right'),
]


# -- Image Source (must be defined before ImageContent) ------------------------


class ImageSource(StrictModel):
    """Image source data for image content."""

    type: Literal['base64']
    media_type: Literal[
        'image/jpeg', 'image/png', 'image/gif', 'image/webp'
    ]  # Anthropic image-source enum (binary-confirmed)
    data: str  # Base64 encoded image data


class ImageContent(StrictModel):
    """Image content block from user messages."""

    type: Literal['image']
    source: ImageSource


class DocumentSource(StrictModel):
    """Document source data for document content (PDFs, etc.)."""

    type: Literal['base64']
    media_type: str  # e.g., 'application/pdf'
    data: str  # Base64 encoded document data


class DocumentContent(StrictModel):
    """Document content block from user messages (PDF uploads, etc.)."""

    type: Literal['document']
    source: DocumentSource


class ToolUseCaller(StrictModel):
    """Caller metadata for tool use (Claude Code 2.1.4+)."""

    type: Literal['direct']  # Only observed value so far


class ToolUseContent(StrictModel):
    """Tool use content block from assistant messages."""

    type: Literal['tool_use']
    id: str
    name: str
    input: ToolInput  # Typed for Claude Code tools, MCPToolInput for MCP tools
    caller: ToolUseCaller | None = None  # Caller metadata (Claude Code 2.1.4+)

    @pydantic.field_validator('input', mode='after')
    @classmethod
    def validate_mcp_tool_fallback(cls, v: ToolInput, info: pydantic.ValidationInfo) -> ToolInput:
        """Enforce that only MCP tools (starting with 'mcp__') can use MCPToolInput fallback.

        All Claude Code built-in tools must have typed models that successfully validate.

        This catches both:
        1. New Claude Code tools we haven't modeled yet
        2. Bugs in existing models (missing/wrong fields causing fallthrough)
        """
        if isinstance(v, MCPToolInput):
            tool_name = info.data.get('name', '')

            # MCP tools are expected to use the fallback - they're third-party
            if tool_name.startswith('mcp__'):
                return v

            # StructuredOutput is a built-in whose input schema is intentionally open
            # (z.object({}).passthrough() in the binary); re-type it from the permissive fallback.
            if tool_name == 'StructuredOutput':
                return StructuredOutputToolInput.model_validate(v.model_dump())

            # ANY Claude Code tool using fallback is a bug - either:
            # - New tool needs a model, OR
            # - Existing model has missing/wrong fields
            raise ValueError(
                f"Claude Code tool '{tool_name}' fell through to MCPToolInput. "
                f'This means either: (1) no typed model exists for this tool, or '
                f'(2) the typed model has missing/incorrect fields. '
                f'Extra fields captured: {list(v.get_extra_fields().keys())}'
            )

        return v


class ToolReferenceContent(StrictModel):
    """Tool reference content block from MCPSearch tool results (Claude Code 2.1.4+).

    Appears in tool_result content when MCPSearch returns matched MCP tools.
    """

    type: Literal['tool_reference']
    tool_name: str  # Full MCP tool name (e.g., "mcp__perplexity__perplexity_research")


# ToolResultContentBlock - for content inside tool_result
ToolResultContentBlock = Annotated[
    TextContent | ImageContent | ToolReferenceContent, pydantic.Field(discriminator='type')
]


class ToolResultContent(StrictModel):
    """Tool result content block from user messages."""

    type: Literal['tool_result']
    tool_use_id: str
    content: str | Sequence[ToolResultContentBlock] | None = (
        None  # Can be string, sequence of content blocks, or missing
    )
    is_error: bool | None = None


# Discriminated union of all message content types
class FallbackModelRef(StrictModel):
    """Model reference inside a FallbackContent from/to pair."""

    model: ModelId


class FallbackContent(StrictModel):
    """Content block marking a mid-turn model fallback (Claude Fable 5 era).

    Binary ``{type:"fallback",from:{model:H.fromModel},to:{model:H.model}}`` -- persisted
    into assistant message content when a turn is served by a fallback model (e.g. a Fable
    refusal retried on Opus). The sibling ``server_fallback`` stream event is never
    persisted. ``from`` is a Python keyword -- aliased per the legitimate-alias rule.
    """

    type: Literal['fallback']
    from_: FallbackModelRef = pydantic.Field(alias='from')
    to: FallbackModelRef


MessageContent = Annotated[
    ThinkingContent
    | TextContent
    | ToolUseContent
    | ToolResultContent
    | ImageContent
    | DocumentContent
    | FallbackContent,
    pydantic.Field(discriminator='type'),
]


# -- Context Management (Claude Code 2.0.51+) ----------------------------------


class ClearThinkingEdit(StrictModel):
    """Applied context edit for clearing thinking blocks."""

    type: Literal['clear_thinking_20251015']
    cleared_thinking_turns: int
    cleared_input_tokens: int


# Union of all applied edit types (add new types here as discovered)
AppliedEdit = ClearThinkingEdit


class ContextManagement(StrictModel):
    """Context management metadata for message responses (Claude Code 2.0.51+)."""

    applied_edits: Sequence[AppliedEdit]  # Can be empty or contain edit records


# -- Message Diagnostics (Claude Code 2.1.119+) --------------------------------


class CacheMissReasonMessagesChanged(StrictModel):
    """Cache miss because messages earlier in the conversation changed.

    Carries the count of input tokens that were not served from cache.
    """

    type: Literal['messages_changed']
    cache_missed_input_tokens: int


class CacheMissReasonModelChanged(StrictModel):
    """Cache miss because the model changed mid-session.

    Carries the count of input tokens that were not served from cache.
    Server-emitted via the cache-diagnosis-2026-04-07 Anthropic API beta
    header, not present in the client binary (consistent with the other
    CacheMissReason variants).
    """

    type: Literal['model_changed']
    cache_missed_input_tokens: int


class CacheMissReasonParamsChanged(StrictModel):
    """Cache miss because request parameters changed.

    Carries the count of input tokens that were not served from cache.
    """

    type: Literal['params_changed']
    cache_missed_input_tokens: int


class CacheMissReasonPreviousMessageNotFound(StrictModel):
    """Cache miss because the previous message couldn't be located in the cache."""

    type: Literal['previous_message_not_found']


class CacheMissReasonSystemChanged(StrictModel):
    """Cache miss because the system prompt changed.

    Carries the count of input tokens that were not served from cache.
    """

    type: Literal['system_changed']
    cache_missed_input_tokens: int


class CacheMissReasonToolsChanged(StrictModel):
    """Cache miss because tool definitions changed.

    Carries the count of input tokens that were not served from cache.
    """

    type: Literal['tools_changed']
    cache_missed_input_tokens: int


class CacheMissReasonUnavailable(StrictModel):
    """Cache miss because cache lookup was unavailable for this request."""

    type: Literal['unavailable']


CacheMissReason = Annotated[
    CacheMissReasonPreviousMessageNotFound
    | CacheMissReasonSystemChanged
    | CacheMissReasonMessagesChanged
    | CacheMissReasonModelChanged
    | CacheMissReasonParamsChanged
    | CacheMissReasonToolsChanged
    | CacheMissReasonUnavailable,
    pydantic.Field(discriminator='type'),
]


class MessageDiagnostics(StrictModel):
    """Diagnostic metadata about request handling (Claude Code 2.1.119+).

    Emitted via the `cache-diagnosis-2026-04-07` Anthropic API beta header.
    Currently always reports prompt-cache outcome. May expand to other
    diagnostic categories in future versions.
    """

    cache_miss_reason: CacheMissReason


# -- Message Structure ---------------------------------------------------------


class Message(StrictModel):
    """A message within a record."""

    role: Literal['user', 'assistant']
    content: Sequence[MessageContent] | str
    # Additional fields that may appear in assistant messages (nested API response)
    type: Literal['message'] | None = pydantic.Field(
        None, description='Message type indicator (present in agent/subprocess responses)'
    )
    model: ModelId | None = pydantic.Field(
        None, description='Claude model identifier (e.g., claude-sonnet-4-5-20250929)'
    )
    id: str | None = pydantic.Field(None, description='Message ID from Claude API')
    stop_reason: (
        Literal['tool_use', 'stop_sequence', 'end_turn', 'refusal', 'max_tokens', 'model_context_window_exceeded']
        | None
    ) = pydantic.Field(None, description='Reason why the model stopped generating')
    stop_sequence: str | None = pydantic.Field(
        None, description='The actual stop sequence string that triggered stopping'
    )
    stop_details: None = pydantic.Field(None, description='Stop details (Claude Code 2.1.81+, always null)')
    usage: TokenUsage | None = pydantic.Field(
        None, description='Token usage information (present in nested API responses)'
    )
    container: None = pydantic.Field(
        None, description='Reserved for future use', json_schema_extra={'status': 'reserved'}
    )
    context_management: ContextManagement | None = pydantic.Field(
        None, description='Context management metadata (Claude Code 2.0.51+)'
    )
    diagnostics: MessageDiagnostics | None = pydantic.Field(
        None,
        description='Cache-miss diagnostics (Claude Code 2.1.119+); null when cache hit or unavailable',
    )


# -- Token Usage ---------------------------------------------------------------


class CacheCreation(StrictModel):
    """Cache creation token breakdown."""

    ephemeral_5m_input_tokens: int
    ephemeral_1h_input_tokens: int


class ServerToolUse(StrictModel):
    """Server-side tool use tracking."""

    web_search_requests: int
    web_fetch_requests: int  # Always present (553/553)


class UsageIteration(StrictModel):
    """A single internal iteration within a turn's token usage (Claude Code 2.1.100+).

    Populated when the turn runs multiple internal inference iterations
    (e.g., adaptive thinking with tool loops). All 6 fields are always present.
    """

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cache_creation: CacheCreation
    type: Literal['fallback_message', 'message']  # last updated 2.1.172
    model: ModelId | None = None  # Model that served this iteration (fallback era; absent on older records)


class OutputTokensDetails(StrictModel):
    """Output-token breakdown (Anthropic API usage passthrough).

    Server-emitted in message.usage; not a client-binary literal (same
    server-diagnostic precedent as CacheMissReason).
    """

    thinking_tokens: int


class TokenUsage(StrictModel):
    """Token usage information for assistant messages."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int  # Always present (115,497/115,497)
    cache_read_input_tokens: int  # Always present (115,497/115,497)
    cache_creation: CacheCreation  # Always present (115,497/115,497)
    service_tier: Literal['standard'] | None = None  # Only value: 'standard' (19018 occurrences) - null for synthetic
    server_tool_use: ServerToolUse | None = None  # Server-side tool use tracking (0.5% present)
    inference_geo: str | None = None  # Inference geography (Claude Code 2.1.31+, e.g. 'not_available')
    iterations: Sequence[UsageIteration] | None = None  # Internal inference iterations (Claude Code 2.1.100+)
    speed: Literal['standard', 'fast'] | None = None  # Speed tier (Claude Code 2.1.41+, 'fast' added 2.1.112+)
    research_preview_2026_02: str | None = None  # Research preview feature flag (e.g. 'active')
    output_tokens_details: OutputTokensDetails | None = None  # Output-token breakdown (API passthrough)


# -- Thinking Metadata ---------------------------------------------------------


def _normalize_ultrathink(v: Any) -> Any:
    """Normalize any casing of 'ultrathink' to lowercase."""
    if isinstance(v, str) and v.lower() == 'ultrathink':
        return 'ultrathink'
    return v  # Let Literal validation fail with original value


class ThinkingTrigger(StrictModel):
    """A thinking trigger with position information."""

    start: int
    end: int
    text: Annotated[
        Literal['ultrathink'], pydantic.BeforeValidator(_normalize_ultrathink)
    ]  # Any casing normalized to lowercase


class ThinkingMetadata(StrictModel):
    """Thinking configuration metadata (full format, pre-2.1.19).

    This is the original format with level/disabled/triggers configuration.
    See also SimpleThinkingMetadata for the newer simplified format.
    """

    level: Literal['none', 'low', 'medium', 'high']  # Strict validation
    disabled: bool
    triggers: Sequence[str | ThinkingTrigger]  # Can be strings or trigger objects


class SimpleThinkingMetadata(StrictModel):
    """Simplified thinking metadata (Claude Code 2.1.19+).

    New format that only specifies max thinking tokens without
    the full level/disabled/triggers configuration.
    See also ThinkingMetadata for the original full format.
    """

    maxThinkingTokens: int


# -- Todo Item -----------------------------------------------------------------


class TodoItem(StrictModel):
    """A single todo item from TodoWrite tool."""

    content: str
    status: Literal['pending', 'in_progress', 'completed']
    activeForm: str


# -- Compact Metadata ----------------------------------------------------------


class CompactMetadata(StrictModel):
    """Metadata for conversation compaction."""

    trigger: Literal['auto', 'manual']  # auto=24, manual=18 across all sessions
    preTokens: int
    preCompactDiscoveredTools: Sequence[str] | None = None  # Tools discovered before compaction (2.1.76+)
    postTokens: int | None = None  # Token count after compaction (Claude Code 2.1.100+)
    durationMs: int | None = None  # Compaction runtime in milliseconds (Claude Code 2.1.100+)
    preservedSegment: PreservedCompactSegment | None = None  # Anchor pointers preserved across compaction
    preservedMessages: PreservedCompactMessages | None = (
        None  # Preserved-message anchors (richer sibling of preservedSegment)
    )


class PreservedCompactMessages(StrictModel):
    """Preserved-message anchors carried across a compaction."""

    anchorUuid: str
    uuids: Sequence[str]
    allUuids: Sequence[str]


class PreservedCompactSegment(StrictModel):
    """Anchor UUIDs identifying the conversation segment preserved across an auto-compact."""

    headUuid: str  # Earliest preserved record
    anchorUuid: str  # Anchor record within the preserved segment
    tailUuid: str  # Most recent preserved record (typically the new logicalParentUuid)


class MicrocompactMetadata(StrictModel):
    """Metadata for micro-compaction (Claude Code 2.1.9+)."""

    trigger: Literal['auto']  # Only 'auto' observed so far
    preTokens: int
    tokensSaved: int
    compactedToolIds: Sequence[str]
    clearedAttachmentUUIDs: EmptySequence  # Only empty arrays observed


# -- API Error -----------------------------------------------------------------


class ApiErrorDetail(StrictModel):
    """Nested API error details."""

    type: Literal['overloaded_error', 'rate_limit_error']
    message: str


class ApiErrorResponse(StrictModel):
    """API error response structure."""

    type: Literal['error']
    error: ApiErrorDetail
    request_id: str | None = None


class ApiError(StrictModel):
    """Complete API error information."""

    status: int
    headers: Mapping[str, str | Sequence[str]]
    requestID: str | None = None  # Can be null for some errors
    error: ApiErrorResponse | None = None  # Can be missing for some errors (e.g., 503)
    type: Literal['overloaded_error', 'rate_limit_error'] | None = None  # Top-level error type (Claude Code 2.1.100+)


# noinspection PyShadowingBuiltins
class ConnectionError(StrictModel):
    """Connection error details for network failures."""

    code: Literal['ConnectionRefused', 'ECONNRESET', 'FailedToOpenSocket']
    path: str  # URL that failed (e.g., "https://api.anthropic.com/v1/messages?beta=true")
    errno: int


class NetworkError(StrictModel):
    """Network error wrapper (for connection failures)."""

    cause: ConnectionError
    type: None = None  # Always null in observed data; reserved/future field mirroring ApiError.type


class EmptyError(StrictModel):
    """Unspecified API error object (Claude Code 2.0.76+).

    Emitted either fields-less (``{}``) or as a bare ``{"type": null}`` -- a
    network-error object whose ``cause`` was undefined at write time. The
    optional null ``type`` mirrors ``NetworkError.type`` / ``ApiError.type``.
    """

    type: None = None  # Always null when present; mirrors NetworkError.type


class NetworkDownConnection(StrictModel):
    """Connection diagnostics for the network-down api_error variant (Claude Code 2.1.161+)."""

    code: Literal['ConnectionRefused', 'ECONNRESET', 'FailedToOpenSocket']
    message: str
    isSSLError: bool


class NetworkDownError(StrictModel):
    """Request-diagnostic api_error envelope (Claude Code 2.1.161+).

    Covers network-down (connection = NetworkDownConnection, isNetworkDown true) and
    request-timeout (connection null, isNetworkDown false) -- the binary builds connection
    via ``VU(H)``, which returns null when there is no socket-level failure. Distinct from
    ApiError (status/headers), NetworkError (cause), EmptyError (null-type-only), and
    RetryableHttpError (has status): identified by formatted + isNetworkDown, no status.
    """

    message: str
    formatted: str
    connection: NetworkDownConnection | None  # null on timeouts (no socket-level failure)
    isNetworkDown: bool
    rateLimits: None = None  # null in this variant


class RateLimitInfo(StrictModel):
    """Rate-limit fields copied from the server ``anthropic-ratelimit-unified-*`` headers.

    ``rateLimitType`` is a pass-through ``str``, NOT a Literal: the producer writes the raw
    header value, so the set is server-defined, not binary-enumerated. The binary only
    *consumes* a known subset (``five_hour``/``seven_day``/``seven_day_opus``/
    ``seven_day_sonnet``) -- per the binary-proven gate, consumer checks don't gate, and a
    Literal would reject server-added types the producer passes through. ``resetsAt`` is a
    unix epoch from the ``-reset`` header.
    """

    rateLimitType: str | None = None
    resetsAt: int | None = None


class RetryableHttpError(StrictModel):
    """Retryable HTTP api_error (429 rate_limit / 529 overloaded) with the network-diagnostic envelope.

    The server responded with a retryable status, so this carries HTTP ``status`` +
    ``requestId`` alongside the same ``formatted``/``connection``/``isNetworkDown``/
    ``rateLimits`` envelope as NetworkDownError. Distinct from ApiError (``headers`` +
    capital-D ``requestID``) and NetworkDownError (no ``status``; ``connection`` is an
    object): identified by required ``status`` + the envelope. ``connection``/``rateLimits``
    are null on a server-side HTTP error (the socket reached the server). Same binary error
    reducer as NetworkDownError. Spans Claude Code 2.1.161+.
    """

    message: str
    status: int
    requestId: str | None = None  # H.requestID ?? undefined -- absent on some records
    formatted: str
    connection: NetworkDownConnection | None  # object on socket failures, null on HTTP errors
    isNetworkDown: bool
    rateLimits: RateLimitInfo | None  # parsed ratelimit headers, else null


# -- MCP Metadata (Claude Code 2.1.19+) ----------------------------------------


class MCPStructuredContent(PermissiveModel):
    """Permissive model for MCP tool structured content (in mcpMeta.structuredContent).

    MCP tools are third-party integrations with varying result schemas. We
    intentionally do not model their specific fields because:
    1. There are 64+ different MCP tools with different result structures
    2. They are user-configured and can change without Claude Code updates
    3. Their schemas are defined by external MCP servers

    NOTE: This is the NEW location for structured MCP content (Claude Code 2.1.19+).
    The same data also appears in tool_result.content as a JSON string. See also
    MCPToolResult which handles the toolUseResult field.

    Uses PermissiveModel to accept any fields while maintaining type observability.
    """


class FastMcpMeta(StrictModel):
    """fastmcp-framework metadata surfaced on MCP tool results (Claude Code 2.1.118+).

    Written by Python MCP servers built on the fastmcp library. Claude Code
    forwards these `_meta.fastmcp` keys through to the session record.
    """

    wrap_result: bool


class McpMetaMetadata(StrictModel):
    """Implementation-framework metadata namespace on MCP tool results.

    Corresponds to the MCP spec's `_meta` field, which servers use to
    attach implementation-specific data. Currently only fastmcp is observed;
    other MCP frameworks (FastMCP-TS, mcp-go, etc.) may add sibling keys here.
    """

    fastmcp: FastMcpMeta | None = None


class McpMeta(StrictModel):
    """MCP tool metadata wrapper (Claude Code 2.1.19+).

    New in 2.1.19: Claude Code extracts structured content from MCP tool
    results that return valid JSON and surfaces it here for easier access.

    DUPLICATION: The same data appears both here (as parsed dict) AND in
    tool_result.content (as JSON string). This is a convenience feature
    for programmatic access without parsing.

    NOT BACKFILLED: Old sessions retain the old format (mcpMeta=None).
    When you resume an old session, new tool calls get mcpMeta but old
    records are not updated. Both patterns can coexist in the same file.

    Only populated for MCP tools returning JSON - tools returning plain text
    (like Perplexity) do not populate this field.

    In 2.1.118+ the `_meta` field is no longer always empty: Claude Code began
    forwarding framework-level metadata (e.g. fastmcp.wrap_result) written
    by the MCP server into the session record.
    """

    meta: McpMetaMetadata | None = pydantic.Field(None, alias='_meta')
    structuredContent: MCPStructuredContent | None = None


# -- File Info -----------------------------------------------------------------


class FileInfo(StrictModel):
    """File information from Read tool (text files)."""

    filePath: PathField
    content: str
    numLines: int
    startLine: int
    totalLines: int


class PdfFileInfo(StrictModel):
    """File information from Read tool (PDF files)."""

    filePath: PathField
    base64: str
    originalSize: int


class ImageFileInfo(StrictModel):
    """File information from Read tool (image files)."""

    type: str  # e.g., 'image/png', 'image/jpeg'
    base64: str
    originalSize: int
    dimensions: ImageDimensions | None = None


class ImageDimensions(StrictModel):
    """Image dimensions."""

    originalWidth: int
    originalHeight: int
    displayWidth: int
    displayHeight: int


# -- Structured Patch ----------------------------------------------------------


class PatchHunk(StrictModel):
    """A single hunk in a git-style patch."""

    oldStart: int
    oldLines: int
    newStart: int
    newLines: int
    lines: Sequence[str]


# -- Tool Use Result Structures ------------------------------------------------


class BashToolResult(StrictModel):
    """Result from Bash tool execution."""

    stdout: str
    stderr: str
    interrupted: bool
    isImage: bool | None = None  # Absent on minimal/early-exit Bash results (Claude Code 2.1.x)
    returnCodeInterpretation: (
        Literal['No matches found', 'Some directories were inaccessible', 'Files differ'] | None
    ) = None
    backgroundTaskId: str | None = None
    backgroundedByUser: bool | None = None  # True if user manually backgrounded with Ctrl+B
    dangerouslyDisableSandbox: bool | None = None  # True if sandbox was disabled for this command
    shellId: str | None = None
    command: str | None = None
    exitCode: int | None = None
    stdoutLines: int | None = None
    stderrLines: int | None = None
    timestamp: str | None = None
    status: Literal['running', 'completed', 'failed'] | None = None
    filterPattern: str | None = None
    noOutputExpected: bool | None = None  # Bash tool hint (Claude Code 2.1.38+)
    persistedOutputPath: PathField | None = None  # Path to persisted large output file (Claude Code 2.1.45+)
    persistedOutputSize: int | None = None  # Size of persisted output in bytes (Claude Code 2.1.45+)
    tokenSaverOutput: str | None = None  # Summarized output for token savings (Claude Code 2.1.80+)
    assistantAutoBackgrounded: bool | None = None  # True if assistant auto-backgrounded the task (2.1.80+)
    staleReadFileStateHint: str | None = None  # Hint when command modified a previously-read file (Claude Code 2.1.x)


class ReadTextToolResult(StrictModel):
    """Result from Read tool execution (text files)."""

    type: Literal['text']
    file: FileInfo


class ReadPdfToolResult(StrictModel):
    """Result from Read tool execution (PDF files)."""

    type: Literal['pdf']
    file: PdfFileInfo


class ReadImageToolResult(StrictModel):
    """Result from Read tool execution (image files)."""

    type: Literal['image']
    file: ImageFileInfo


class ReadPartsFileInfo(StrictModel):
    """File info for Read tool 'parts' variant — PDF split into per-page tool-result files."""

    filePath: PathField
    originalSize: int
    outputDir: PathField
    count: int


class ReadPartsToolResult(StrictModel):
    """Result from Read tool execution (large PDF split into parts) (Claude Code 2.1.x)."""

    type: Literal['parts']
    file: ReadPartsFileInfo


class ReadFileUnchangedFileInfo(StrictModel):
    """File info for Read tool 'file_unchanged' variant — file content unchanged since last read."""

    filePath: PathField


class ReadFileUnchangedToolResult(StrictModel):
    """Result from Read tool when file is unchanged since last read (Claude Code 2.1.x)."""

    type: Literal['file_unchanged']
    file: ReadFileUnchangedFileInfo


class GlobToolResult(StrictModel):
    """Result from Glob/file search tool execution."""

    mode: Literal['files_with_matches'] | None = None
    filenames: Sequence[str]
    numFiles: int
    durationMs: int | None = None
    truncated: bool | None = None
    appliedLimit: int | None = None  # Limit that was applied to results


class GrepToolResult(StrictModel):
    """Result from Grep/content search tool execution."""

    mode: Literal['content', 'count', 'files_with_matches']
    numFiles: int
    filenames: Sequence[str]
    content: str | None = None
    numLines: int | None = None
    numMatches: int | None = None
    appliedLimit: int | None = None
    appliedOffset: int | None = None  # Offset applied to results (Claude Code 2.1.38+)


class EditToolResult(StrictModel):
    """Result from Edit tool execution."""

    filePath: PathField
    oldString: str
    newString: str
    originalFile: str | None  # Null when Claude Code didn't capture pre-edit contents (Claude Code 2.1.x)
    userModified: bool
    replaceAll: bool
    structuredPatch: Sequence[PatchHunk]


class WriteToolResult(StrictModel):
    """Result from Write tool execution."""

    type: Literal['create', 'update']
    filePath: PathField
    content: str
    structuredPatch: Sequence[PatchHunk] | None = None
    originalFile: str | None = None  # Present but always None for 'create' type
    userModified: bool | None = None  # Whether user edited the content before accepting (Claude Code 2.1.x)


class TodoToolResult(StrictModel):
    """Result from TodoWrite tool execution."""

    oldTodos: Sequence[TodoItem]
    newTodos: Sequence[TodoItem]


class TaskToolStats(StrictModel):
    """Per-subagent tool-use statistics returned with TaskToolResult (Claude Code 2.1.x)."""

    readCount: int
    searchCount: int
    bashCount: int
    editFileCount: int
    linesAdded: int
    linesRemoved: int
    otherToolCount: int


class TaskToolResult(StrictModel):
    """Result from Task/agent tool execution."""

    status: Literal['completed']
    prompt: str
    content: Sequence[MessageContent]
    totalDurationMs: int
    totalTokens: int
    totalToolUseCount: int
    usage: TokenUsage
    agentId: str  # Always present (621/621)
    agentType: str | None = None  # Subagent type (e.g., 'Explore', 'general-purpose') (Claude Code 2.1.x)
    toolStats: TaskToolStats | None = None  # Per-subagent tool-use stats (Claude Code 2.1.x)


# -- TaskCreate/TaskUpdate/TaskList Result Structures (Claude Code 2.1.17+) ----


class TaskListItem(StrictModel):
    """A task item in TaskList results."""

    id: str
    subject: str
    status: Literal['pending', 'in_progress', 'completed']
    blockedBy: Sequence[str]
    owner: str | None = None  # Only present when task is assigned


class TaskListToolResult(StrictModel):
    """Result from TaskList tool - list of all tasks."""

    tasks: Sequence[TaskListItem]


class TaskSingleItem(StrictModel):
    """A task item in TaskCreate/TaskUpdate results (minimal fields)."""

    id: str
    subject: str


class TaskSingleToolResult(StrictModel):
    """Result from TaskCreate/TaskUpdate - single task confirmation."""

    task: TaskSingleItem


class StatusChange(StrictModel):
    """Status transition details."""

    from_: str = pydantic.Field(validation_alias=pydantic.AliasChoices('from', 'from_'))
    to: str


class TaskUpdateSuccessResult(StrictModel):
    """Alternative result format from TaskUpdate (success-based)."""

    success: bool
    taskId: str
    updatedFields: Sequence[str]
    statusChange: str | StatusChange | None = None
    error: str | None = None  # Present when update fails
    verificationNudgeNeeded: bool | None = None  # Hint to verify task completion (2.1.80+)


class TaskGetItem(StrictModel):
    """A task item in TaskGet results (full detail)."""

    id: str
    subject: str
    description: str
    status: Literal['pending', 'in_progress', 'completed']
    blocks: Sequence[str]
    blockedBy: Sequence[str]


class TaskGetToolResult(StrictModel):
    """Result from TaskGet tool - single task with full details."""

    task: TaskGetItem


class LSPToolResult(StrictModel):
    """Result from LSP tool execution."""

    operation: str  # e.g., 'definition', 'references'
    result: str
    filePath: PathField
    resultCount: int
    fileCount: int


# -- Task Model (Session Artifact - On-Disk Format) ----------------------------
#
# Task files are session-scoped artifacts stored separately from session JSONL.
# They don't appear in session records but are persisted under the session ID.
#
# Path: ~/.claude/tasks/{session_id}/{id}.json


class Task(StrictModel):
    """Canonical task model (on-disk format).

    Path: ~/.claude/tasks/{session_id}/{id}.json
    """

    id: str  # Matches filename without .json
    subject: str  # Brief imperative title (e.g., "Run tests")
    description: str  # Detailed what-to-do with acceptance criteria
    activeForm: str | None = None  # Present continuous for UI spinner (e.g., "Running tests")
    status: Literal['pending', 'in_progress', 'completed']
    blocks: Sequence[str]  # Task IDs that cannot start until this completes
    blockedBy: Sequence[str]  # Task IDs that must complete before this starts
    owner: str | None = None  # Agent/owner identifier


# -- AskUserQuestion Structures ------------------------------------------------


class QuestionOption(StrictModel):
    """A single option in a user question."""

    label: str
    description: str
    markdown: str | None = None  # Markdown preview of the option (Claude Code 2.1.50+)
    preview: str | None = None  # Preview text for the option (alternative to markdown)


class UserQuestion(StrictModel):
    """A question to ask the user."""

    question: str
    header: str
    options: Sequence[QuestionOption]
    multiSelect: bool = False  # Binary: v.boolean().default(!1) — optional with default


class QuestionAnnotation(StrictModel):
    """Annotation on a user's answer to a question (Claude Code 2.1.45+)."""

    markdown: str | None = None  # Markdown preview content of the selected option
    notes: str | None = None  # Free-text notes the user added to their selection


class AskUserQuestionToolResult(StrictModel):
    """Result from AskUserQuestion tool execution."""

    questions: Sequence[UserQuestion]
    answers: Mapping[str, str]  # Mapping of question text to user's answer
    annotations: Mapping[str, QuestionAnnotation] | None = None  # Per-question annotations (Claude Code 2.1.45+)


# -- WebSearch Structures ------------------------------------------------------


class WebSearchResult(StrictModel):
    """A single web search result."""

    title: str
    url: str


class WebSearchToolResult(StrictModel):
    """Result from WebSearch tool execution."""

    query: str
    results: Sequence[WebSearchResult] | str  # Can be list of results or string explanation
    durationSeconds: int | float  # Can be int or float depending on timing


class WebFetchToolResult(StrictModel):
    """Result from WebFetch tool execution."""

    url: str
    code: int
    codeText: str
    bytes: int
    result: str
    durationMs: int


class ExitPlanModeToolResult(StrictModel):
    """Result from ExitPlanMode tool execution."""

    plan: str | None  # Can be null when plan is in external file
    isAgent: bool
    filePath: PathField | None = None  # Plan file path (present in newer versions)


class McpResource(StrictModel):
    """A single MCP resource from ListMcpResourcesTool."""

    name: str
    title: str | None = None  # Optional - some resources don't have title
    uri: str
    description: str
    mimeType: str
    server: str


# NOTE: ListMcpResourcesTool returns a bare list[McpResource] (not wrapped in a dict)
# This is handled by UserRecord.toolUseResult: list[McpResource] variant


class McpResourceContent(StrictModel):
    """A single resource content item from ReadMcpResourceTool."""

    uri: str
    mimeType: str
    text: str


class ReadMcpResourceToolResult(StrictModel):
    """Result from ReadMcpResourceTool execution."""

    contents: Sequence[McpResourceContent]


class KillShellToolResult(StrictModel):
    """Result from KillShell tool execution."""

    success: bool
    shellId: str


class TaskStopToolResult(StrictModel):
    """Result from TaskStop tool execution."""

    message: str
    task_id: str
    task_type: Literal['local_bash', 'local_agent']
    command: str | None = None  # Shell command (for local_bash tasks, Claude Code 2.1.38+)


class ToolSearchToolResult(StrictModel):
    """Result from ToolSearch tool execution."""

    matches: Sequence[str]
    query: str
    # Field name changed over versions - data has one or the other
    total_deferred_tools: int | None = None
    total_mcp_tools: int | None = None
    pending_mcp_servers: Sequence[str] | None = None  # Servers still connecting (Claude Code 2.1.51+)


class BashOutputToolResult(StrictModel):
    """Result from BashOutput tool (background shell output).

    Different from BashToolResult: requires shellId/command/status/exitCode
    but does NOT have interrupted/isImage fields.
    """

    shellId: str
    command: str
    status: Literal['running', 'completed', 'failed', 'killed']
    exitCode: int | None  # null when status='running'
    stdout: str
    stderr: str
    stdoutLines: int
    stderrLines: int
    timestamp: str
    filterPattern: str | None = None


# -- TaskOutput Polling Results ------------------------------------------------


class BackgroundTask(StrictModel):
    """Background task state from TaskOutput tool polling."""

    task_id: str
    task_type: Literal['local_bash', 'local_agent']
    status: Literal['running', 'completed', 'failed', 'killed']
    description: str
    output: str
    exitCode: int | None = None  # For bash tasks, null when running/killed
    prompt: str | None = None  # For agent tasks
    result: str | None = None  # For agent tasks
    error: str | None = None  # For failed agent tasks


class TaskOutputPollingResult(StrictModel):
    """Result from TaskOutput tool - polling background task state."""

    retrieval_status: Literal['not_ready', 'success', 'timeout']
    task: BackgroundTask | None  # null when retrieval_status='timeout'


# -- Async Task Launch Results -------------------------------------------------


class AsyncTaskLaunchResult(StrictModel):
    """Result from launching async Task (with or without output file tracking)."""

    isAsync: Literal[True]
    status: Literal['async_launched']
    agentId: str
    description: str
    prompt: str
    outputFile: str | None = None  # Path to output file (sometimes missing)
    canReadOutputFile: bool | None = None  # Whether the output file can be read (2.1.47+)


# -- Multi-Agent Retrieval Results ---------------------------------------------


class AgentState(StrictModel):
    """State of a background agent (may be running or completed)."""

    status: Literal['running', 'completed', 'failed']
    description: str
    prompt: str
    result: str | None = None  # null when still running


class AgentsRetrievalResult(StrictModel):
    """Result from retrieving multiple agent states."""

    retrieval_status: Literal['not_ready', 'success', 'timeout']
    agents: Mapping[str, AgentState]  # Empty dict when not_ready


# -- KillShell Message Variant -------------------------------------------------


class KillShellMessageResult(StrictModel):
    """Alternative KillShell result with message format (snake_case shell_id)."""

    message: str  # e.g., "Successfully killed shell: b18fae0 (...)"
    shell_id: str  # Note: uses snake_case, not camelCase


# -- WebSearch Nested Structure Variant ----------------------------------------


class WebSearchResultWrapper(StrictModel):
    """Wrapper for web search results with tool use ID (nested structure variant)."""

    tool_use_id: str
    content: Sequence[WebSearchResult]


class WebSearchNestedResult(StrictModel):
    """Result from WebSearch tool with nested structure variant."""

    query: str
    results: Sequence[WebSearchResultWrapper | str]  # Can be wrapper or text
    durationSeconds: float


# -- Handoff Command Result ----------------------------------------------------


class HandoffCommandResult(StrictModel):
    """Result from handoff command execution."""

    success: Literal[True]
    commandName: Literal['handoff']
    allowedTools: Sequence[str]


# -- EnterPlanMode Tool Result -------------------------------------------------


class EnterPlanModeToolResult(StrictModel):
    """Result from EnterPlanMode tool execution."""

    message: str  # Plan mode entry confirmation


class SkillToolResult(StrictModel):
    """Result from Skill tool execution."""

    success: bool
    commandName: str  # Skill name (e.g. 'canvas-design')
    allowedTools: Sequence[str] | None = None  # Tool allow-list granted by the skill (Claude Code 2.1.x)


# -- EnterWorktree Tool Result (Claude Code 2.1.63+) ---------------------------


class EnterWorktreeToolResult(StrictModel):
    """Result from EnterWorktree tool - confirms worktree creation."""

    worktreePath: str  # Absolute path to the created worktree
    worktreeBranch: str  # Git branch name for the worktree
    message: str  # Confirmation message


# -- ExitWorktree Tool Result (Claude Code 2.1.x) ------------------------------


class ExitWorktreeToolResult(StrictModel):
    """Result from ExitWorktree tool - confirms worktree exit (keep or remove)."""

    action: Literal['keep', 'remove']
    originalCwd: PathField  # Directory to return to after exiting the worktree
    worktreePath: PathField  # Absolute path of the exited worktree
    worktreeBranch: str  # Git branch name of the exited worktree
    message: str  # Confirmation message
    discardedFiles: int | None = None  # Count of uncommitted files discarded (action='remove' only)
    discardedCommits: int | None = None  # Count of commits discarded (action='remove' only)


# -- CronCreate Tool Result (Claude Code 2.1.x) --------------------------------


class CronCreateToolResult(StrictModel):
    """Result from CronCreate tool - confirms scheduled task creation."""

    id: str  # Job ID (e.g., '194d2a8f') used to cancel or inspect the schedule
    humanSchedule: str  # Human-readable schedule (e.g., '* * * * *')
    recurring: bool  # Whether the task fires on every cron match
    durable: bool  # Whether the task survives Claude Code restart


# -- TeamCreate Tool Result (Claude Code 2.1.63+) ------------------------------


class TeamCreateToolResult(StrictModel):
    """Result from TeamCreate tool - confirms team creation."""

    team_name: str  # Name of the created team
    team_file_path: str  # Path to team config file
    lead_agent_id: str  # ID of the team lead agent


# -- Agent Teammate Spawned Result (Claude Code 2.1.63+) -----------------------


class AgentTeammateSpawnedResult(StrictModel):
    """Result from Agent tool when spawning a teammate in team mode."""

    status: Literal['teammate_spawned']
    prompt: str  # Instructions sent to the spawned teammate
    teammate_id: str  # ID of the spawned teammate
    agent_id: str  # Agent ID (same as teammate_id for team agents)
    agent_type: str  # e.g., 'general-purpose'
    model: str  # Model used by the spawned agent
    name: str  # Human-readable name of the teammate
    color: str  # Terminal color for the teammate
    tmux_session_name: str  # tmux session hosting the agent
    tmux_window_name: str  # tmux window hosting the agent
    tmux_pane_id: str  # tmux pane ID for the agent
    team_name: str  # Team the agent belongs to
    is_splitpane: bool  # Whether the agent runs in a split pane
    plan_mode_required: bool  # Whether plan mode is required for this agent


# -- SendMessage Tool Result (Claude Code 2.1.63+) -----------------------------


class SendMessageRouting(StrictModel):
    """Routing metadata for a sent message between team members."""

    sender: str  # Name of the sending agent
    senderColor: str | None = None  # Terminal color of sender (absent for team-lead)
    target: str  # Recipient (prefixed with @)
    targetColor: str | None = None  # Terminal color of target (absent when sender is teammate)
    summary: str  # Brief summary of the message
    content: str  # Full message content


class SendMessageToolResult(StrictModel):
    """Result from SendMessage tool - confirms message delivery between teammates."""

    success: bool
    message: str  # Delivery confirmation (e.g., "Message sent to X's inbox")
    routing: SendMessageRouting


# -- MCP Tool Result (Third-Party Tools) ---------------------------------------


class MCPToolResult(PermissiveModel):
    """Permissive model for MCP (Model Context Protocol) tool results.

    MCP tools are third-party integrations with varying result schemas. We
    intentionally do not model their specific fields because:
    1. There are 64+ different MCP tools with different result structures
    2. They are user-configured and can change without Claude Code updates
    3. Their schemas are defined by external MCP servers

    NOTE: Unlike MCPToolInput, we cannot validate that only MCP tools use this
    fallback. Tool results don't carry the tool name (it's in the previous
    assistant message's ToolUseContent). Observability is provided by
    find_fallbacks() in validate_models.py instead.

    Uses PermissiveModel to accept any fields while maintaining type observability.
    """


# Union of all tool result types (validated left-to-right, most specific first)
# NOTE: Order matters! More specific models (more required fields) should come first.
# NOTE: Unlike ToolInput, there's no validator enforcing MCP-only fallback here.
# This is intentional: tool results don't carry the tool name (it's in the previous
# assistant message's ToolUseContent), so we can't easily distinguish MCP vs Claude Code.
# Observability is provided by find_fallbacks() in validate_models.py instead.
ToolResult = Annotated[
    # Core tool results (most specific first)
    BashToolResult  # Bash tool (requires interrupted/isImage)
    | BashOutputToolResult  # BashOutput tool (background shell, no interrupted/isImage)
    | ReadTextToolResult  # Read text files
    | ReadPdfToolResult  # Read PDF files
    | ReadImageToolResult  # Read image files
    | ReadPartsToolResult  # Read large PDF split into parts (Claude Code 2.1.x)
    | ReadFileUnchangedToolResult  # Read when file unchanged since last read (Claude Code 2.1.x)
    | EditToolResult
    | WriteToolResult
    | GrepToolResult
    | GlobToolResult
    | TodoToolResult
    | TaskToolResult  # Completed sync task
    | TaskOutputPollingResult  # Polling async task
    | AsyncTaskLaunchResult  # Launched async task
    | AgentsRetrievalResult  # Multi-agent polling
    | TaskListToolResult  # TaskList result (2.1.17+)
    | TaskGetToolResult  # TaskGet result - full task details (2.1.50+)
    | TaskSingleToolResult  # TaskCreate/TaskUpdate result (2.1.17+)
    | TaskUpdateSuccessResult  # Alternative TaskUpdate result (success-based)
    | TaskStopToolResult  # TaskStop result (2.1.25+)
    | ToolSearchToolResult  # ToolSearch/MCPSearch result (2.1.4+)
    | LSPToolResult  # LSP tool result (definition, references, etc.)
    | AskUserQuestionToolResult
    | WebSearchNestedResult  # Nested structure variant (more specific - has tool_use_id in results)
    | WebSearchToolResult  # Simple structure
    | WebFetchToolResult
    | ExitPlanModeToolResult
    | EnterPlanModeToolResult  # Plan mode entry
    | KillShellMessageResult  # Message variant (has message + shell_id)
    | KillShellToolResult  # Original variant (has success + shellId)
    | ReadMcpResourceToolResult  # ReadMcpResourceTool result
    | HandoffCommandResult  # Handoff command
    | SkillToolResult  # Skill tool result
    | ExitWorktreeToolResult  # ExitWorktree result (2.1.x) - must precede EnterWorktree (superset of required fields)
    | EnterWorktreeToolResult  # EnterWorktree result (2.1.63+)
    | CronCreateToolResult  # CronCreate result (2.1.x)
    | TeamCreateToolResult  # TeamCreate result (2.1.63+)
    | AgentTeammateSpawnedResult  # Agent teammate spawned result (2.1.63+)
    | SendMessageToolResult  # SendMessage result (2.1.63+)
    | MCPToolResult,  # Fallback for MCP tools (PermissiveModel for observability)
    pydantic.Field(union_mode='left_to_right'),
]

# -- Base Record ---------------------------------------------------------------


class BaseRecord(StrictModel):
    """Base class for all session record types."""

    type: str
    uuid: str
    timestamp: str
    sessionId: str
    forkedFrom: ForkOrigin | None = (
        None  # Source session/message when this record belongs to a forked session (Claude Code 2.1.8+)
    )
    # Session origin kind; absent for interactive sessions (the omitted default). Binary enum is
    # interactive|bg|daemon|daemon-worker (from CLAUDE_CODE_SESSION_KIND); only 'bg' observed in JSONL.
    sessionKind: Literal['bg'] | None = None


class ForkOrigin(StrictModel):
    """Backpointer to the source session/message a forked session branched from."""

    sessionId: str
    messageUuid: str


# -- User Record ---------------------------------------------------------------


class UserRecordOrigin(StrictModel):
    """Origin metadata shared by UserRecord.origin and QueuedCommandAttachment.origin (2.1.87+).

    kind is the closed origin-producer set. 'human' gained a real write site in 2.1.172
    (resume-background-agent path); before that it was consumer-guard-only.
    """

    kind: Literal[
        'auto-continuation', 'channel', 'coordinator', 'human', 'peer', 'task-notification'
    ]  # last updated 2.1.172


class UserRecord(BaseRecord):
    """User message record."""

    type: Literal['user']
    cwd: PathField
    parentUuid: str | None
    isSidechain: bool
    userType: Literal['external']
    version: CCVersionStrField
    gitBranch: str
    message: Message
    origin: UserRecordOrigin | None = None  # Message origin metadata (Claude Code 2.1.87+)
    projectPaths: PathListField | None = pydantic.Field(
        None, description='Additional project paths beyond cwd (each path will be translated)'
    )
    budgetTokens: int | None = pydantic.Field(None, description='Token budget limit for this request')
    skills: None = pydantic.Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    mcp: None = pydantic.Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})
    agentId: str | None = pydantic.Field(
        None, description='Agent ID for subprocess/agent records (references agent-{agentId}.jsonl)'
    )
    # --- Message visibility tiers ---
    # Two flags control where a message appears:
    #   Fully visible (default):     shown in terminal UI, sent to Claude API, saved to JSONL
    #   isMeta=True:                 hidden from terminal, sent to Claude API, saved to JSONL
    #   isVisibleInTranscriptOnly:   hidden from terminal, saved to JSONL only (archival record)
    isMeta: bool | None = pydantic.Field(
        None,
        description='UI visibility flag: true = sent to Claude API but hidden from user terminal. '
        'Primary source: skill/command content injection (the full SKILL.md content Claude acts on). '
        'Also: local-command caveats, auto-resume prompts, system context injection. '
        'Absent/false = visible in terminal.',
    )
    thinkingMetadata: ThinkingMetadata | SimpleThinkingMetadata | None = pydantic.Field(
        None, description='Extended thinking configuration (Claude 3.7+, simplified format in 2.1.19+)'
    )
    isVisibleInTranscriptOnly: bool | None = pydantic.Field(
        None,
        description='Archival record: saved to JSONL but hidden from terminal UI. '
        'Claude Code manages API inclusion during context reconstruction (e.g., on session resume). '
        'Only observed with isCompactSummary=True (compact summary UserRecords).',
    )
    isCompactSummary: bool | None = pydantic.Field(
        None,
        description='Marks this as a compact summary. Always paired with isVisibleInTranscriptOnly=True. '
        'The two-record compaction pattern: compact_boundary system record followed immediately by '
        'a UserRecord with this flag containing the summary text. '
        'This replaced the standalone SummaryRecord mechanism.',
    )
    toolUseResult: Annotated[
        Sequence[ToolResultContentBlock]  # TextContent/ImageContent with 'type' discriminator - must come first
        | Sequence[McpResource]  # MCP resources (no 'type' field, has 'name', 'uri', etc.)
        | ToolResult
        | str
        | None,
        pydantic.Field(union_mode='left_to_right'),
    ] = None  # Tool execution metadata (validated left-to-right)
    todos: Sequence[TodoItem] | None = pydantic.Field(None, description='Todo list state (Claude Code 2.0.47+)')
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    imagePasteIds: Sequence[int] | None = pydantic.Field(None, description='IDs of pasted images in this message')
    sourceToolUseID: str | None = pydantic.Field(
        None,
        description='Tool use ID that generated this message (e.g., toolu_015eZkLKZz5JVkGC1zZrnnKm) (Claude Code 2.0.76+)',
    )
    sourceToolAssistantUUID: str | None = pydantic.Field(
        None,
        description='UUID of the assistant message that created the tool use this record responds to',
    )
    permissionMode: Literal['default', 'acceptEdits', 'plan', 'bypassPermissions', 'auto'] | None = pydantic.Field(
        None, description='Permission mode for the request (Claude Code 2.1.15+)'
    )
    planContent: str | None = pydantic.Field(None, description='Plan content for plan mode submissions')
    mcpMeta: McpMeta | None = pydantic.Field(
        None, description='MCP tool structured content metadata (Claude Code 2.1.19+)'
    )
    promptId: str | None = pydantic.Field(None, description='Prompt identifier (Claude Code 2.1.74+)')
    promptSource: Literal['queued', 'sdk', 'system', 'typed'] | None = pydantic.Field(
        None,
        description="Prompt origin (Claude Code 2.1.161+): non-interactive→'sdk', isMeta→'system', "
        "queued-replay→'queued', else 'typed'.",
    )
    entrypoint: str | None = pydantic.Field(None, description='Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)')
    teamName: str | None = pydantic.Field(None, description='Team name when running in multi-agent team mode')
    agentName: str | None = pydantic.Field(None, description='Agent name within team (may be absent for lead agent)')
    interruptedMessageId: str | None = pydantic.Field(
        None,
        description='API message ID (msg_*) of the assistant message interrupted by this user '
        'input (Claude Code 2.1.149+).',
    )


# -- Assistant Record ----------------------------------------------------------


class AssistantRecord(BaseRecord):
    """Assistant message record."""

    type: Literal['assistant']
    cwd: PathField
    parentUuid: str | None = pydantic.Field(..., description='UUID of parent record (null for root agent records)')
    message: Message
    # Note: usage/stopReason are optional for agent records (nested in message instead)
    usage: TokenUsage | None = pydantic.Field(
        None, description='Token usage for this request (null for agent records - usage in message instead)'
    )
    stopReason: None = pydantic.Field(
        None, description='Reserved for future use', json_schema_extra={'status': 'reserved'}
    )
    model: ModelId | None = pydantic.Field(
        None, description='Claude model identifier (null for agent records - model in message instead)'
    )
    requestDuration: int | None = pydantic.Field(None, description='Request duration in milliseconds')
    requestId: str | None = pydantic.Field(None, description='Claude API request ID')
    agentId: str | None = pydantic.Field(
        None, description='Agent ID for subprocess/agent records (references agent-{agentId}.jsonl)'
    )
    isSidechain: bool | None = pydantic.Field(
        None, description='Indicates sidechain/subprocess execution (present in agent records)'
    )
    userType: str | None = pydantic.Field(None, description='User type (present in agent records)')
    version: CCVersionStrField | None = pydantic.Field(
        None, description='Claude Code version (present in agent records)'
    )
    gitBranch: str | None = pydantic.Field(None, description='Git branch (present in agent records)')
    isApiErrorMessage: bool | None = pydantic.Field(None, description='Indicates this message represents an API error')
    apiError: Literal['max_output_tokens'] | None = pydantic.Field(
        None, description='API error code (Claude Code 2.1.15+)'
    )
    error: (
        Literal['authentication_failed', 'invalid_request', 'model_not_found', 'rate_limit', 'server_error', 'unknown']
        | None
    ) = pydantic.Field(None, description='Error type for API error messages')  # last updated ≤2.1.172
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    entrypoint: str | None = pydantic.Field(None, description='Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)')
    teamName: str | None = pydantic.Field(None, description='Team name when running in multi-agent team mode')
    agentName: str | None = pydantic.Field(None, description='Agent name within team')
    errorDetails: str | None = pydantic.Field(None, description='API error details string (e.g., prompt too long)')
    apiErrorStatus: int | None = pydantic.Field(
        None, description='HTTP status code for API errors (e.g., 400, 529) (Claude Code 2.1.100+)'
    )
    attributionAgent: str | None = pydantic.Field(
        None,
        description='Agent type attribution for assistant records emitted under a specific agent '
        '(Claude Code 2.1.121+); references the running agent name (e.g., "unrestricted-worker"). '
        'Typed str (not Literal) since values come from the user agent registry, matching the '
        'agentName/teamName precedent.',
    )
    attributionSkill: str | None = pydantic.Field(
        None,
        description='Active skill name (e.g., "recover-session"); mirrors attributionAgent.',
    )
    attributionPlugin: str | None = pydantic.Field(
        None,
        description='Active plugin name (e.g., "codex"); paired with attributionSkill when '
        'the assistant turn is emitted by a plugin-installed skill (Claude Code 2.1.121+).',
    )
    attributionMcpServer: str | None = pydantic.Field(
        None,
        description='MCP server name attributed to an assistant turn that invoked an MCP tool '
        '(e.g., "selenium-browser"); paired with attributionMcpTool (Claude Code 2.1.149+).',
    )
    attributionMcpTool: str | None = pydantic.Field(
        None,
        description='MCP tool name attributed to the turn (e.g., "navigate"); paired with '
        'attributionMcpServer (Claude Code 2.1.149+).',
    )


# -- Summary Record (does NOT inherit from BaseRecord - different schema) ------


class SummaryRecord(StrictModel):
    """Session summary record (minimal schema, no uuid/timestamp).

    NOTE: Zero occurrences observed in recent sessions (50+ files, 521K+ records).
    Compaction now uses UserRecord with isCompactSummary=True instead.
    This type may be from an earlier Claude Code version. Kept for backward compatibility.
    """

    type: Literal['summary']
    summary: str
    leafUuid: str


# -- System Record -------------------------------------------------------------


class SystemRecord(BaseRecord):
    """System message record (standard system messages)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    systemType: str
    message: str  # Always string (0 dict occurrences across all sessions)


# -- System Subtype Records (discriminated by subtype field) -------------------


class LocalCommandSystemRecord(BaseRecord):
    """System record for local command output (subtype=local_command)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['local_command']
    content: str  # Command output XML
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool  # UI visibility flag: true = internal plumbing hidden from terminal
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None


class CompactBoundarySystemRecord(BaseRecord):
    """System record for conversation compaction (subtype=compact_boundary)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['compact_boundary']
    content: str  # e.g., "Conversation compacted"
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None
    logicalParentUuid: str | None = None
    compactMetadata: CompactMetadata | None = None


class MicrocompactBoundarySystemRecord(BaseRecord):
    """System record for micro-compaction (subtype=microcompact_boundary, Claude Code 2.1.9+)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['microcompact_boundary']
    content: Literal['Context microcompacted']
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None
    microcompactMetadata: MicrocompactMetadata


class ApiErrorSystemRecord(BaseRecord):
    """System record for API errors (subtype=api_error)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['api_error']
    level: Literal['error', 'warning'] | None = None
    isSidechain: bool | None = None  # Optional for api_error
    userType: str | None = None  # Optional for api_error
    version: CCVersionStrField | None = None  # Optional for api_error
    gitBranch: str | None = None  # Optional for api_error
    slug: str | None = pydantic.Field(None, description='Human-readable session slug (Claude Code 2.0.51+)')
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None
    cause: ConnectionError | None = None  # Connection error details (for network failures)
    error: (
        ApiError | NetworkError | NetworkDownError | RetryableHttpError | EmptyError
    )  # api/network/network-down/retryable-http/empty (EmptyError last — no required fields)
    retryInMs: float
    retryAttempt: int
    maxRetries: int


class InformationalSystemRecord(BaseRecord):
    """System record for informational messages (subtype=informational)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['informational']
    content: str | None = None
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isMeta: bool | None = None
    isSidechain: bool | None = None
    userType: str | None = None
    version: CCVersionStrField | None = None
    gitBranch: str | None = None
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    slug: str | None = None
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None


class TurnDurationSystemRecord(BaseRecord):
    """System record for turn duration tracking (subtype=turn_duration, Claude Code 2.1.1+)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['turn_duration']
    durationMs: int  # Duration of the turn in milliseconds
    messageCount: int | None = None  # Number of messages in the turn (Claude Code 2.1.87+)
    pendingBackgroundAgentCount: int | None = None  # Background agents still running at turn end (Claude Code 2.1.152+)
    pendingWorkflowCount: int | None = (
        None  # Workflows still running at turn end (sibling of pendingBackgroundAgentCount)
    )
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None


class HookInfo(StrictModel):
    """Information about a hook execution."""

    command: str
    durationMs: int | None = None  # Hook duration ms (Claude Code 2.1.120+; absent on pre-2.1.120 records)


class StopHookSummarySystemRecord(BaseRecord):
    """System record for stop hook summary (subtype=stop_hook_summary, Claude Code 2.1.14+)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['stop_hook_summary']
    hookCount: int
    hookInfos: Sequence[HookInfo]
    hookErrors: Sequence[str]  # Empty sequence observed so far
    hookAdditionalContext: Sequence[str] | None = None  # Stop/SubagentStop additionalContext (2.1.163+)
    preventedContinuation: bool
    stopReason: str  # Can be empty string
    hasOutput: bool
    level: Literal['info', 'error', 'warning', 'suggestion'] | None = None
    isSidechain: bool | None = None
    userType: str | None = None
    version: CCVersionStrField | None = None
    gitBranch: str | None = None
    toolUseID: str | None = None  # Tool use ID if triggered by tool
    slug: str | None = None  # Human-readable session slug
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    teamName: str | None = None
    agentName: str | None = None


class BridgeStatusSystemRecord(BaseRecord):
    """System record for remote-control bridge status (subtype=bridge_status, Claude Code 2.1.51+)."""

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['bridge_status']
    content: str  # e.g., "/remote-control is active. Code in CLI or at https://claude.ai/code/..."
    url: str  # claude.ai/code session URL
    upgradeNudge: str | None = None  # Mobile app upgrade prompt (Claude Code 2.1.87+)
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    agentId: str | None = None  # Agent identifier on sidechain records (Claude Code 2.1.x)
    teamName: str | None = None
    agentName: str | None = None


class ScheduledTaskFireSystemRecord(BaseRecord):
    """System record for scheduled task execution (subtype=scheduled_task_fire, Claude Code 2.1.85+).

    Emitted as a transcript marker when /loop or CronCreate tasks fire.
    """

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['scheduled_task_fire']
    content: str  # e.g., "Running scheduled task (Apr 7 7:23am)"
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None
    agentId: str | None = None  # Present on sidechain subagent records (Claude Code 2.1.112+)
    teamName: str | None = None
    agentName: str | None = None


class AwaySummarySystemRecord(BaseRecord):
    """System record for session recap (subtype=away_summary, Claude Code 2.1.108+).

    Emitted when returning to an idle session to summarize prior goal/state.
    Controlled by `/recap` or `CLAUDE_CODE_ENABLE_AWAY_SUMMARY`.
    """

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['away_summary']
    content: str  # Recap text summarizing session goal and current state
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None


class AgentsKilledSystemRecord(BaseRecord):
    """System record emitted when running subagents are killed (subtype=agents_killed, Claude Code 2.1.139+).

    Carries no payload beyond the base envelope; signals subagent termination
    (e.g., on Esc / session interrupt).
    """

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['agents_killed']
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None


class ModelRefusalFallbackSystemRecord(BaseRecord):
    """System record for an API-side model refusal triggering a fallback retry (subtype=model_refusal_fallback).

    Binary ``{type:"system",subtype:"model_refusal_fallback",direction:"retry",content,
    level:"warning",trigger:"refusal",originalModel,fallbackModel,requestId,
    apiRefusalCategory,apiRefusalExplanation,retractedMessageUuids?}``. Producer present
    since ≤2.1.162; surfaced in JSONL by Claude Fable 5 refusal fallbacks (2.1.170+).
    originalModel/fallbackModel carry config-level ids including the ``[1m]`` context
    suffix -- plain str, not ModelId.
    """

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['model_refusal_fallback']
    content: str
    level: Literal['warning']
    direction: Literal['retry']
    trigger: Literal['refusal']  # last updated 2.1.172
    originalModel: str
    fallbackModel: str
    requestId: str | None = None  # Passthrough; key absent when the fallback event has no request id
    apiRefusalCategory: str | None = None
    apiRefusalExplanation: str | None = None
    retractedMessageUuids: Sequence[str] | None = None  # Second write site only (refusal-retract path)
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None


class ModelFallbackSystemRecord(BaseRecord):
    """System record for a model-availability fallback (subtype=model_fallback).

    Binary yields ``{type:"system",subtype:"model_fallback",content:`Switched to ...`,
    level:"warning",trigger,originalModel,fallbackModel}`` when the configured model is
    unavailable (the fallbackModel setting); the if-guard restricts trigger to exactly
    model_not_found | permission_denied. Producer present since ≤2.1.162. No
    requestId/direction/apiRefusal* -- those belong to ModelRefusalFallbackSystemRecord.
    """

    type: Literal['system']
    cwd: PathField
    parentUuid: str | None
    subtype: Literal['model_fallback']
    content: str
    level: Literal['warning']
    trigger: Literal['model_not_found', 'permission_denied']  # last updated 2.1.172
    originalModel: str
    fallbackModel: str
    isMeta: bool
    isSidechain: bool
    userType: str
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None


# Union of system subtype records
SystemSubtypeRecord = Annotated[
    LocalCommandSystemRecord
    | CompactBoundarySystemRecord
    | MicrocompactBoundarySystemRecord
    | ApiErrorSystemRecord
    | InformationalSystemRecord
    | TurnDurationSystemRecord
    | StopHookSummarySystemRecord
    | BridgeStatusSystemRecord
    | ScheduledTaskFireSystemRecord
    | AwaySummarySystemRecord
    | AgentsKilledSystemRecord
    | ModelRefusalFallbackSystemRecord
    | ModelFallbackSystemRecord,
    pydantic.Field(discriminator='subtype'),
]


# -- File History Snapshot Record (does NOT inherit from BaseRecord - different schema) ---


class FileBackupInfo(StrictModel):
    """Backup information for a tracked file."""

    backupFileName: str | None  # Can be null for first version
    version: int
    backupTime: str


class FileHistorySnapshot(StrictModel):
    """Snapshot data for file history tracking."""

    messageId: str
    trackedFileBackups: Mapping[str, FileBackupInfo]  # Map of file paths to backup data
    timestamp: str


class FileHistorySnapshotRecord(StrictModel):
    """File history snapshot record (different schema from regular records)."""

    type: Literal['file-history-snapshot']
    messageId: str
    snapshot: FileHistorySnapshot
    isSnapshotUpdate: bool


# -- Queue Operation Record (does NOT inherit from BaseRecord - no uuid field) ---


class QueueOperationRecord(StrictModel):
    """Queue operation record (minimal schema, no uuid)."""

    type: Literal['queue-operation']
    operation: Literal['enqueue', 'dequeue', 'remove', 'popAll']  # Queue operation type
    timestamp: str
    sessionId: str
    content: str | Sequence[MessageContent] | None = pydantic.Field(
        None, description='User input content for the queued operation (string or structured message)'
    )
    data: None = pydantic.Field(None, description='Reserved for future use', json_schema_extra={'status': 'reserved'})


# -- Custom Title Record (does NOT inherit from BaseRecord - minimal schema) ---


class CustomTitleRecord(StrictModel):
    """Custom title record for user-defined session names (minimal schema, no uuid)."""

    type: Literal['custom-title']
    customTitle: str  # User-defined session title
    sessionId: str


# -- Progress Record (Claude Code 2.1.9+) --------------------------------------


class HookProgressData(StrictModel):
    """Progress data for hook execution."""

    type: Literal['hook_progress']
    hookEvent: str  # e.g., "SessionStart"
    hookName: str  # e.g., "SessionStart:startup"
    command: PathField


class McpProgressStartedData(StrictModel):
    """Progress data for MCP tool start."""

    type: Literal['mcp_progress']
    status: Literal['started']
    serverName: str
    toolName: str


class McpProgressCompletedData(StrictModel):
    """Progress data for MCP tool completion."""

    type: Literal['mcp_progress']
    status: Literal['completed']
    serverName: str
    toolName: str
    elapsedTimeMs: int


class McpProgressFailedData(StrictModel):
    """Progress data for MCP tool failure."""

    type: Literal['mcp_progress']
    status: Literal['failed']
    serverName: str
    toolName: str
    elapsedTimeMs: int


class BashProgressData(StrictModel):
    """Progress data for bash command execution."""

    type: Literal['bash_progress']
    output: str
    fullOutput: str
    elapsedTimeSeconds: int
    totalLines: int
    totalBytes: int | None = None  # Byte count of output (Claude Code 2.1.45+)
    taskId: str | None = None  # Background task ID (Claude Code 2.1.45+)
    timeoutMs: int | None = None  # Present when command has explicit timeout


class WaitingForTaskData(StrictModel):
    """Progress data for waiting on a task."""

    type: Literal['waiting_for_task']
    taskDescription: str
    taskType: Literal['local_bash', 'local_agent']


class SearchResultsReceivedData(StrictModel):
    """Progress data for web search results received (Claude Code 2.1.9+)."""

    type: Literal['search_results_received']
    resultCount: int
    query: str


class QueryUpdateData(StrictModel):
    """Progress data for search query update (Claude Code 2.1.9+)."""

    type: Literal['query_update']
    query: str


class AgentProgressData(StrictModel):
    """Progress data for agent/subagent execution."""

    type: Literal['agent_progress']
    agentId: str
    prompt: str
    message: Mapping[str, Any]  # strict_typing_linter.py: loose-typing — API error body is unstructured JSON
    normalizedMessages: Sequence[Mapping[str, Any]] | None = (
        None  # strict_typing_linter.py: loose-typing — API error body is unstructured JSON
    )
    resume: str | None = None  # Agent resume ID (Claude Code 2.1.15+)


# Discriminated union of progress data types
# NOTE: Models with more required fields must come first (left_to_right matching)
ProgressData = Annotated[
    AgentProgressData  # Most fields (5 required)
    | HookProgressData
    | McpProgressCompletedData
    | McpProgressFailedData
    | McpProgressStartedData
    | BashProgressData
    | WaitingForTaskData
    | SearchResultsReceivedData  # Web search progress (Claude Code 2.1.9+)
    | QueryUpdateData,  # Search query update (fewest fields - must be last)
    pydantic.Field(union_mode='left_to_right'),
]


class ProgressRecord(StrictModel):
    """Progress record for tracking long-running operations (Claude Code 2.1.9+)."""

    type: Literal['progress']
    uuid: str
    timestamp: str
    sessionId: str
    cwd: PathField
    parentUuid: str | None
    isSidechain: bool
    userType: Literal['external']
    version: CCVersionStrField
    gitBranch: str
    data: ProgressData
    parentToolUseID: str
    toolUseID: str
    slug: str | None = None  # Missing on first record before slug assigned
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)
    agentId: str | None = None  # Present in agent subfiles (references agent-{agentId}.jsonl)
    teamName: str | None = pydantic.Field(None, description='Team name when running in multi-agent team mode')
    agentName: str | None = pydantic.Field(None, description='Agent name within team')


# -- PR Link Record (Claude Code 2.1.31+) --------------------------------------


class PrLinkRecord(StrictModel):
    """PR link record created when Claude Code opens a pull request."""

    type: Literal['pr-link']
    sessionId: str
    timestamp: str
    prNumber: int
    prUrl: str
    prRepository: str


# -- Saved Hook Context Record (Claude Code 2.1.27+) ---------------------------


class SavedHookContextRecord(StrictModel):
    """Saved hook context record for persisting hook output across session boundaries."""

    type: Literal['saved_hook_context']
    uuid: str
    timestamp: str
    sessionId: str
    cwd: PathField
    parentUuid: str | None
    isSidechain: bool
    userType: Literal['external']
    version: CCVersionStrField
    gitBranch: str
    content: Sequence[str]
    hookName: str
    toolUseID: str
    hookEvent: str
    entrypoint: str | None = None  # Client entrypoint (e.g., "cli") (Claude Code 2.1.80+)


# -- Agent Name Record (Claude Code 2.1.80+) -----------------------------------


class AgentNameRecord(StrictModel):
    """Records the human-readable name assigned to a worker agent session.

    Written to main session files to associate agent sessions with descriptive names.
    """

    type: Literal['agent-name']
    agentName: str
    sessionId: str


# -- AI Title Record (Claude Code 2.1.122+) ------------------------------------


class AiTitleRecord(StrictModel):
    """AI-generated session title shown in the `/resume` UI."""

    type: Literal['ai-title']
    aiTitle: str
    sessionId: str


# -- Last Prompt Record (Claude Code 2.1.69+) ----------------------------------


class LastPromptRecord(StrictModel):
    """Records the user's last prompt text.

    Typically the final record in a session file. Claude Code 2.1.120+ also writes a
    placeholder near session start with `lastPrompt` absent until the first user input.
    Claude Code 2.1.121+ adds `leafUuid` referencing the latest message UUID in this
    conversation branch (parallels SummaryRecord.leafUuid).
    """

    type: Literal['last-prompt']
    lastPrompt: str | None = None
    sessionId: str
    leafUuid: str | None = None


# -- Worktree State Record (Claude Code 2.1.81+) ------------------------------


class WorktreeSessionData(StrictModel):
    """Worktree session metadata embedded in WorktreeStateRecord."""

    originalCwd: str
    worktreePath: str
    worktreeName: str
    worktreeBranch: str | None = None  # Absent when entering an existing worktree — no branch created (2.1.105+)
    sessionId: str
    originalBranch: str | None = None  # Absent when entering existing worktree (2.1.105+)
    originalHeadCommit: str | None = None  # Absent when entering existing worktree (2.1.105+)
    enteredExisting: bool | None = None  # True when EnterWorktree(path=...) attaches to existing worktree (2.1.105+)


class WorktreeStateRecord(StrictModel):
    """Records worktree state at the start of a worktree session (line 1 only).

    Does NOT inherit from BaseRecord -- minimal schema with no uuid/timestamp.
    """

    type: Literal['worktree-state']
    worktreeSession: WorktreeSessionData | None  # null on exit-worktree state records
    sessionId: str


# -- Permission Mode Record (Claude Code 2.1.90+) -----------------------------


class PermissionModeRecord(StrictModel):
    """Records the active permission mode for a session.

    Does NOT inherit from BaseRecord -- minimal schema with no uuid/timestamp.
    """

    type: Literal['permission-mode']
    permissionMode: Literal['default', 'acceptEdits', 'plan', 'bypassPermissions', 'auto']
    sessionId: str


# -- Session Mode Record (Claude Code 2.1.69+) --------------------------------


class SessionModeRecord(StrictModel):
    """Records the active session mode for a session.

    Resumption-state sidecar written alongside PermissionModeRecord; does NOT
    inherit from BaseRecord -- minimal schema with no uuid/timestamp. Binary
    ``$.push({type:"mode",mode:this.currentSessionMode,sessionId:q})`` where mode is
    ``isCoordinatorMode() ? "coordinator" : "normal"`` (same projection as
    IsolationLatchRecord.side).
    """

    type: Literal['mode']
    mode: Literal['coordinator', 'normal']  # last updated ≤2.1.156
    sessionId: str


# -- Bridge Session Record -----------------------------------------------------


class BridgeSessionRecord(StrictModel):
    """Records the remote-control bridge association for a session.

    Resumption-state sidecar (no uuid/timestamp), written alongside
    PermissionModeRecord. Binary: ``$.push({type:"bridge-session",sessionId:q,
    bridgeSessionId:...,lastSequenceNum:...})``; emitted by /remote-control
    bridge sessions. Distinct from BridgeStatusSystemRecord (a system subtype).
    """

    type: Literal['bridge-session']
    sessionId: str
    bridgeSessionId: str
    lastSequenceNum: int


# -- Agent/session-state records (binary-proven; producers present since ≤2.1.156) -----
#
# Minimal records the binary writes via appendEntry/appendFile and re-emits on session-
# file rotation (reAppendSessionMetadata). Each docstring cites its binary write site;
# none carry the uuid/timestamp/cwd envelope unless noted. Five sibling rT4 routing types
# are intentionally NOT modeled -- see the deferred-types note after this section.


class TagRecord(StrictModel):
    """User-assigned session tag.

    Binary rich path ``append([{type:"tag",tag,sessionId,uuid,timestamp}])`` (set-tag
    handler); the metadata-reemit path ``$.push({type:"tag",tag,sessionId})`` omits
    uuid/timestamp. ``tag`` is ``""`` when cleared. Distinct from the slash-command
    parser's ``{type:"tag",path,push,...}``, which is not a record.
    """

    type: Literal['tag']
    tag: str
    sessionId: str
    uuid: str | None = None  # Rich set-tag write path only
    timestamp: str | None = None  # Rich set-tag write path only


class AgentColorRecord(StrictModel):
    """Palette color assigned to a worker-agent session.

    Binary ``JzH(sessionFile,{type:"agent-color",agentColor,sessionId})``; written only
    when the color is truthy (``"default"`` is skipped).
    """

    type: Literal['agent-color']
    agentColor: str
    sessionId: str


class AgentSettingRecord(StrictModel):
    """Agent-type setting for a session.

    Binary metadata-reemit ``$.push({type:"agent-setting",agentSetting,sessionId})``;
    ``agentSetting`` is the agentType identifier string (e.g. ``"custom"``), not an object.
    """

    type: Literal['agent-setting']
    agentSetting: str
    sessionId: str


class IsolationLatchRecord(StrictModel):
    """Coordinator/worker isolation side for a session.

    Binary ``Y$4(sessionFile,{type:"isolation-latch",side,sessionId})`` where
    ``side = isCoordinatorMode() ? "coordinator" : "normal"`` (same projection as
    SessionModeRecord.mode).
    """

    type: Literal['isolation-latch']
    side: Literal['coordinator', 'normal']  # last updated ≤2.1.156
    sessionId: str


class SpeculationAcceptRecord(StrictModel):
    """Records a speculative-decoding acceptance and the wall-time it saved.

    Binary ``appendFile(sessionFile,{type:"speculation-accept",timestamp,timeSavedMs})``,
    written only when ``timeSavedMs > 0``. No sessionId/uuid envelope.
    """

    type: Literal['speculation-accept']
    timestamp: str
    timeSavedMs: int


class ContentReplacement(StrictModel):
    """A single tool-result replacement within a ContentReplacementRecord."""

    kind: Literal['tool-result']
    toolUseId: str
    replacement: str


class ContentReplacementRecord(StrictModel):
    """Replaces persisted tool-result content (compaction/fork dedup).

    Binary has three write sites: bulk ``{type,sessionId,replacements}``; fork/clone adds
    ``uuid``+``timestamp``; per-agent (route-by-agent) adds ``agentId``. Each replacement
    swaps a tool_use's stored output for a compacted form.
    """

    type: Literal['content-replacement']
    sessionId: str
    replacements: Sequence[ContentReplacement]
    agentId: str | None = None  # Per-agent (route-by-agent) write path only
    uuid: str | None = None  # Fork/clone write path only
    timestamp: str | None = None  # Fork/clone write path only


class ForkContextRefRecord(StrictModel):
    """Subagent fork-context pointer linking a forked subagent journal to its parent.

    Binary ``appendEntry({type:"fork-context-ref",agentId,parentSessionId,parentLastUuid,
    contextLength})`` (route-by-agent -> ``subagents/agent-*.jsonl``). Minimal pointer:
    no uuid/timestamp/sessionId/cwd envelope -- ``parentSessionId``/``parentLastUuid``
    reference the parent instead.
    """

    type: Literal['fork-context-ref']
    agentId: str
    parentSessionId: str
    parentLastUuid: str
    contextLength: int  # Parent-message COUNT forked into the subagent (binary g.length), not a token count


# Deferred rT4 routing types (registered but not modeled -- binary-proven gate unmet):
#   frame-link            -- routing entry only; no producer ever existed, and the routing
#                            key itself was removed in 2.1.172.
#   marble-origami-{commit,snapshot,reset} -- dead code (producers present, zero callers
#                            in 2.1.162/2.1.163; "context-collapse" feature unshipped);
#                            spread payload unenumerable.
#   attribution-snapshot  -- genuine writer, but payload is built behind the module export
#                            boundary (only {type, messageId} provable).
# Model each when its feature ships and a real record appears in JSONL (strict validation
# surfaces it via SessionRecordError).


# -- Attachment Record (Claude Code 2.1.90+) -----------------------------------


class CompanionIntroAttachment(StrictModel):
    """Companion creature introduction (/buddy feature)."""

    type: Literal['companion_intro']
    name: str
    species: str


class McpInstructionsDeltaAttachment(StrictModel):
    """MCP server instruction changes (added/removed tool descriptions)."""

    type: Literal['mcp_instructions_delta']
    addedNames: Sequence[str]
    addedBlocks: Sequence[str]
    removedNames: Sequence[str]


class DeferredToolsDeltaAttachment(StrictModel):
    """Deferred-tool registration changes for the tool-search system (Claude Code 2.1.120+)."""

    type: Literal['deferred_tools_delta']
    addedNames: Sequence[str]
    addedLines: Sequence[str]
    removedNames: Sequence[str]
    readdedNames: Sequence[str] | None = None  # Tools that were removed and then re-registered (2.1.128+)
    pendingMcpServers: Sequence[str] | None = None  # MCP servers still connecting at emit time (2.1.128+)


class AgentListingDeltaAttachment(StrictModel):
    """Agent registry changes — added/removed agent types for the Agent tool (Claude Code 2.1.90+)."""

    type: Literal['agent_listing_delta']
    addedTypes: Sequence[str]
    addedLines: Sequence[str]
    removedTypes: Sequence[str]
    isInitial: bool
    showConcurrencyNote: bool


class TaskReminderItem(StrictModel):
    """Task entry within a task_reminder attachment."""

    id: str
    subject: str
    description: str
    status: Literal['pending', 'in_progress', 'completed']
    blocks: Sequence[str]
    blockedBy: Sequence[str]
    activeForm: str | None = None  # Only set for tasks that were given an active-form label
    owner: str | None = None  # Task owner (e.g. 'frontend', 'backend') when set via TaskCreate
    metadata: Mapping[str, Any] | None = (
        None  # strict_typing_linter.py: loose-typing — binary schema is v.record(v.string(), v.unknown()).optional()
    )


class TaskReminderAttachment(StrictModel):
    """Task list reminder injected into the conversation (Claude Code 2.1.x)."""

    type: Literal['task_reminder']
    content: Sequence[TaskReminderItem]
    itemCount: int


class HookSuccessAttachment(StrictModel):
    """Hook executed successfully — result injected as session attachment."""

    type: Literal['hook_success']
    hookName: str
    toolUseID: str
    hookEvent: str
    content: str
    stdout: str
    stderr: str
    exitCode: int
    command: str
    durationMs: int


class HookBlockingErrorData(StrictModel):
    """Nested error detail for hook_blocking_error."""

    blockingError: str
    command: str


class HookBlockingErrorAttachment(StrictModel):
    """Hook returned a blocking error — tool execution was halted."""

    type: Literal['hook_blocking_error']
    hookName: str
    toolUseID: str
    hookEvent: str
    blockingError: HookBlockingErrorData


class HookNonBlockingErrorAttachment(StrictModel):
    """Hook returned a non-blocking error — tool continued."""

    type: Literal['hook_non_blocking_error']
    hookName: str
    toolUseID: str
    hookEvent: str
    stderr: str
    stdout: str
    exitCode: int
    command: str
    durationMs: int


class HookCancelledAttachment(StrictModel):
    """Hook execution was cancelled (e.g., aborted before completion)."""

    type: Literal['hook_cancelled']
    hookName: str
    toolUseID: str
    hookEvent: str
    command: str
    durationMs: int


class HookAdditionalContextAttachment(StrictModel):
    """Hook returned additional context to inject into the conversation."""

    type: Literal['hook_additional_context']
    content: str | Sequence[str]
    hookName: str
    toolUseID: str
    hookEvent: str


class QueuedCommandAttachment(StrictModel):
    """User command queued while Claude was working. prompt may be str or content blocks."""

    type: Literal['queued_command']
    prompt: str | Sequence[TextContent | ImageContent]
    commandMode: Literal['plan', 'prompt', 'task-notification']  # last updated ≤2.1.156
    imagePasteIds: Sequence[int] | None = None  # Present when prompt contains pasted images
    source_uuid: str | None = pydantic.Field(
        None,
        description='UUID of the source message that triggered this queued command. '
        'Rare (1/355 globally observed); only seen with commandMode="prompt". '
        'Set by the SDK replay path (Claude Code 2.1.20+ — `SDKUserMessageReplay` events '
        'when `replayUserMessages` is enabled). Note: snake_case in the wire schema, '
        'anomalous vs surrounding camelCase fields.',
    )
    origin: UserRecordOrigin | None = pydantic.Field(
        None, description='Queued-command origin (Claude Code 2.1.157+); see UserRecordOrigin for the kind set.'
    )


class DynamicSkillAttachment(StrictModel):
    """Skill discovered dynamically at a non-default location (Claude Code 2.1.x)."""

    type: Literal['dynamic_skill']
    skillDir: PathField
    skillNames: Sequence[str]
    displayPath: str


class SkillListingAttachment(StrictModel):
    """Periodic listing of available skills (Claude Code 2.1.x)."""

    type: Literal['skill_listing']
    content: str
    skillCount: int
    isInitial: bool
    names: Sequence[str] | None = None  # Skill identifiers in the listing (optional)


class InvokedSkill(StrictModel):
    """A skill invocation record inside an InvokedSkillsAttachment."""

    name: str
    path: str  # e.g., 'userSettings:review-pr-comments' (source:slug)
    content: str


class InvokedSkillsAttachment(StrictModel):
    """Skills that were actually invoked for this turn (Claude Code 2.1.119+).

    Distinct from dynamic_skill (discovery of a skill dir under cwd) and
    skill_listing (periodic catalog of available skills): this attachment is
    emitted when the user invokes one or more skills via /<skill-name>, and
    carries the full skill content that was injected into the conversation.
    """

    type: Literal['invoked_skills']
    skills: Sequence[InvokedSkill]


class NestedMemoryContent(StrictModel):
    """Nested CLAUDE.md content payload."""

    path: PathField
    type: str  # e.g., "Project"
    content: str
    contentDiffersFromDisk: bool | None = None
    rawContent: str | None = None


class NestedMemoryAttachment(StrictModel):
    """Nested CLAUDE.md / memory file discovered under cwd."""

    type: Literal['nested_memory']
    path: PathField
    content: NestedMemoryContent
    displayPath: str


class FileAttachmentFileContent(StrictModel):
    """File content payload nested inside FileAttachment.content."""

    filePath: PathField
    content: str
    numLines: int
    startLine: int
    totalLines: int


class FileAttachmentContent(StrictModel):
    """Content wrapper for FileAttachment — carries the underlying text file."""

    type: Literal['text']
    file: FileAttachmentFileContent


class FileAttachment(StrictModel):
    """Generic file reference attached to the conversation."""

    type: Literal['file']
    filename: PathField
    content: FileAttachmentContent
    displayPath: str


class AlreadyReadFileAttachment(StrictModel):
    """Reminder injected when Claude re-Reads an unchanged file in the same session.

    Carries the cached snapshot the model has already seen so it can skip re-issuing
    the Read tool. Shape mirrors FileAttachment exactly (filename + displayPath +
    wrapped FileAttachmentContent), so the nested types are reused directly.
    """

    type: Literal['already_read_file']
    filename: PathField
    content: FileAttachmentContent
    displayPath: str


class EditedTextFileAttachment(StrictModel):
    """File edited externally — snippet with line numbers shown to Claude.

    displayPath is rare (~2% of observed records, only on more recent
    Claude Code versions when the file is outside cwd); modeled optional.
    """

    type: Literal['edited_text_file']
    filename: PathField
    snippet: str
    displayPath: str | None = None


class OpenedFileInIdeAttachment(StrictModel):
    """File opened in the connected IDE (VSCode/JetBrains)."""

    type: Literal['opened_file_in_ide']
    filename: PathField
    displayPath: str | None = None  # cwd-relative path; present on a minority of records


class SelectedLinesInIdeAttachment(StrictModel):
    """Lines selected in the connected IDE."""

    type: Literal['selected_lines_in_ide']
    ideName: str
    lineStart: int
    lineEnd: int
    filename: PathField
    content: str
    displayPath: str


class DiagnosticPosition(StrictModel):
    """LSP-style position (line/character, 0-based)."""

    line: int
    character: int


class DiagnosticSeverityRange(StrictModel):
    """LSP-style position range for a diagnostic."""

    start: DiagnosticPosition
    end: DiagnosticPosition


class DiagnosticItem(StrictModel):
    """A single LSP-style diagnostic."""

    message: str
    severity: str  # 'Error', 'Warning', 'Info', 'Hint'
    range: DiagnosticSeverityRange


class DiagnosticFile(StrictModel):
    """Diagnostics grouped by file URI."""

    uri: str
    diagnostics: Sequence[DiagnosticItem]


class DiagnosticsAttachment(StrictModel):
    """IDE diagnostics injected into the conversation."""

    type: Literal['diagnostics']
    files: Sequence[DiagnosticFile]
    isNew: bool


class PlanFileReferenceAttachment(StrictModel):
    """Plan file content referenced in the conversation."""

    type: Literal['plan_file_reference']
    planFilePath: PathField
    planContent: str


class PlanModeExitAttachment(StrictModel):
    """User exited plan mode — recorded whether the plan file was written."""

    type: Literal['plan_mode_exit']
    planFilePath: PathField
    planExists: bool | None = None


class PlanModeAttachment(StrictModel):
    """Plan mode reminder — injected while plan mode is active (Claude Code 2.1.112+)."""

    type: Literal['plan_mode']
    reminderType: str
    isSubAgent: bool
    planFilePath: PathField
    planExists: bool


class PlanModeReentryAttachment(StrictModel):
    """User re-entered plan mode with a previously saved plan (Claude Code 2.1.112+)."""

    type: Literal['plan_mode_reentry']
    planFilePath: PathField


class CompactFileReferenceAttachment(StrictModel):
    """File reference injected after a compact operation."""

    type: Literal['compact_file_reference']
    filename: PathField
    displayPath: str


class CommandPermissionsAttachment(StrictModel):
    """Tool allow-list applied by a slash command."""

    type: Literal['command_permissions']
    allowedTools: Sequence[str]


class DateChangeAttachment(StrictModel):
    """Date rolled over mid-session — injected so Claude has the current date."""

    type: Literal['date_change']
    newDate: str


class AutoModeAttachment(StrictModel):
    """Auto mode activation reminder."""

    type: Literal['auto_mode']
    reminderType: str


class AutoModeExitAttachment(StrictModel):
    """Auto mode exited."""

    type: Literal['auto_mode_exit']


class WorkflowKeywordRequestAttachment(StrictModel):
    """Workflow-trigger keyword detected in the prompt (Claude Code 2.1.153+).

    Marker injected when a prompt's "workflow" keyword may trigger a dynamic
    workflow; carries no payload beyond the type tag.
    """

    type: Literal['workflow_keyword_request']


class UltrathinkEffortAttachment(StrictModel):
    """Ultrathink effort-level indicator for the turn (Claude Code 2.1.139+).

    Marker injected when an ultrathink trigger raises the turn's thinking
    effort; carries no payload beyond the type tag.
    """

    type: Literal['ultrathink_effort']


class TaskStatusAttachment(StrictModel):
    """Background-execution status delta (the ``claude agents`` / Agent-tool runtime).

    A "task" here is a background job (``taskType`` local_agent | local_bash; for an agent
    ``taskId`` IS the agentId) -- the runtime sense, NOT the TaskCreate/TaskUpdate TODO
    tracker (tell them apart by status: TODO is pending/in_progress/completed; runtime is
    queued/running/completed/failed/killed). Binary ``{type:"task_status",taskId,taskType,
    status,description,deltaSummary,outputFilePath}``; ``deltaSummary`` is the running
    progress summary else the error (nullable). ``status`` is binary-enumerated (literal
    write sites), unlike server pass-throughs like RateLimitInfo.rateLimitType.
    """

    type: Literal['task_status']
    taskId: str
    taskType: Literal['local_agent', 'local_bash']
    status: Literal['completed', 'failed', 'killed', 'queued', 'running']
    description: str
    deltaSummary: str | None  # progress summary while running, else error; nullable
    outputFilePath: str


AttachmentData = Annotated[
    CompanionIntroAttachment
    | McpInstructionsDeltaAttachment
    | DeferredToolsDeltaAttachment
    | AgentListingDeltaAttachment
    | TaskReminderAttachment
    | HookSuccessAttachment
    | HookBlockingErrorAttachment
    | HookNonBlockingErrorAttachment
    | HookCancelledAttachment
    | HookAdditionalContextAttachment
    | QueuedCommandAttachment
    | DynamicSkillAttachment
    | SkillListingAttachment
    | InvokedSkillsAttachment
    | NestedMemoryAttachment
    | FileAttachment
    | AlreadyReadFileAttachment
    | EditedTextFileAttachment
    | OpenedFileInIdeAttachment
    | SelectedLinesInIdeAttachment
    | DiagnosticsAttachment
    | PlanFileReferenceAttachment
    | PlanModeAttachment
    | PlanModeExitAttachment
    | PlanModeReentryAttachment
    | CompactFileReferenceAttachment
    | CommandPermissionsAttachment
    | DateChangeAttachment
    | AutoModeAttachment
    | AutoModeExitAttachment
    | WorkflowKeywordRequestAttachment
    | UltrathinkEffortAttachment
    | TaskStatusAttachment,
    pydantic.Field(discriminator='type'),
]


class AttachmentRecord(BaseRecord):
    """Attachment record for non-message content injected into sessions.

    Shares BaseRecord fields plus UserRecord metadata fields. Distinguished
    from UserRecord by having an 'attachment' field instead of 'message'.
    """

    type: Literal['attachment']
    cwd: PathField
    parentUuid: str | None  # Null on the first record of a session
    isSidechain: bool
    userType: Literal['external']
    version: CCVersionStrField
    gitBranch: str
    slug: str | None = None
    entrypoint: str | None = None
    agentId: str | None = None
    teamName: str | None = None
    agentName: str | None = None
    attachment: AttachmentData


# -- Session Record (Discriminated Union) --------------------------------------

# Union of all record types (validated left-to-right)
# NOTE: Cannot use discriminator='type' because multiple records have type='system'
# System subtype records must come before SystemRecord so they match first
SessionRecord = Annotated[
    UserRecord
    | AssistantRecord
    | SummaryRecord
    | LocalCommandSystemRecord  # Must be before SystemRecord!
    | CompactBoundarySystemRecord  # Must be before SystemRecord!
    | MicrocompactBoundarySystemRecord  # Must be before SystemRecord!
    | ApiErrorSystemRecord  # Must be before SystemRecord!
    | InformationalSystemRecord  # Must be before SystemRecord!
    | TurnDurationSystemRecord  # Must be before SystemRecord!
    | StopHookSummarySystemRecord  # Must be before SystemRecord!
    | BridgeStatusSystemRecord  # Must be before SystemRecord!
    | ScheduledTaskFireSystemRecord  # Must be before SystemRecord!
    | AwaySummarySystemRecord  # Must be before SystemRecord!
    | AgentsKilledSystemRecord  # Must be before SystemRecord!
    | ModelRefusalFallbackSystemRecord  # Must be before SystemRecord!
    | ModelFallbackSystemRecord  # Must be before SystemRecord!
    | SystemRecord
    | FileHistorySnapshotRecord
    | QueueOperationRecord
    | CustomTitleRecord
    | ProgressRecord
    | PrLinkRecord
    | SavedHookContextRecord
    | AgentNameRecord
    | AiTitleRecord
    | LastPromptRecord
    | WorktreeStateRecord
    | PermissionModeRecord
    | SessionModeRecord
    | BridgeSessionRecord
    | TagRecord
    | AgentColorRecord
    | AgentSettingRecord
    | IsolationLatchRecord
    | SpeculationAcceptRecord
    | ContentReplacementRecord
    | ForkContextRefRecord
    | AttachmentRecord
    | BaseRecord,  # Fallback for unknown system subtypes
    pydantic.Field(union_mode='left_to_right'),
]

# Type adapter for validating session records (required for union types)
SessionRecordAdapter: pydantic.TypeAdapter[SessionRecord] = pydantic.TypeAdapter(SessionRecord)


# -- Session Metadata ----------------------------------------------------------


class SessionMetadata(StrictModel):
    """Metadata extracted from a session."""

    session_id: str
    record_count: int
    first_timestamp: str
    last_timestamp: str
    unique_cwds: Sequence[str]
    unique_project_paths: Sequence[str]
    user_message_count: int
    assistant_message_count: int
    summary_count: int
    system_message_count: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_creation_tokens: int
    total_cache_read_tokens: int
    models_used: Sequence[str]
    tools_used: Sequence[str]
    files_touched: Sequence[str]
    git_branches: Sequence[str]


# -- Analysis Results ----------------------------------------------------------


class SessionAnalysis(StrictModel):
    """Complete analysis of a session."""

    metadata: SessionMetadata
    summary_text: str | None = None
    cost_estimate_usd: float | None = None
    duration_seconds: float | None = None


# -- Workflow Run-Journal (subagents/workflows/wf_*/journal.jsonl) -------------
#
# Claude Code's Workflow tool writes a per-run resume/cache journal: a `started`
# entry when an agent begins, a `result` entry when its result is recorded.
# Structurally distinct from transcript records — no uuid/timestamp/sessionId.
# `result` is the agent's return value, agent-defined and unbounded.


class WorkflowJournalStarted(StrictModel):
    """Run-journal entry written when a workflow agent begins."""

    type: Literal['started']
    key: str  # Content-addressed cache key: "v2:" + sha256 hex
    agentId: str  # The agent this entry concerns; matches sibling agent-<agentId>.jsonl


class WorkflowJournalResult(StrictModel):
    """Run-journal entry written when a workflow agent's result is recorded."""

    type: Literal['result']
    key: str  # Same key as the paired "started" entry
    agentId: str  # As above; the runtime may record it as '' (empty)
    result: Any  # strict_typing_linter.py: loose-typing — agent-defined return value, no fixed shape


WorkflowJournalRecord = Annotated[
    WorkflowJournalStarted | WorkflowJournalResult,
    pydantic.Field(discriminator='type'),
]


# -- Utility Functions ---------------------------------------------------------


# noinspection PyNewStyleGenericSyntax
def validated_copy[T: pydantic.BaseModel](
    model: T, update: Mapping[str, Any]
) -> T:  # strict_typing_linter.py: loose-typing — generic copy accepts any field values
    """Create a validated copy of a model with updates.

    This is the type-safe way to modify frozen Pydantic models.
    The returned instance is validated through the model's __init__.

    Args:
        model: The source model instance
        update: Fields to update (keys must be valid field names)

    Returns:
        A new validated instance of the same type with updates applied

    Example:
        updated_record = validated_copy(record, {'sessionId': new_session_id})
    """
    model_class = type(model)
    new_data = model.model_dump()
    new_data.update(update)
    return model_class(**new_data)


# -- Fast Dispatch Validation --------------------------------------------------

# Per-type TypeAdapters bypass the SessionRecord left-to-right union scan.
# When adding a new record type to SessionRecord, also add a branch below.
_user_adapter = pydantic.TypeAdapter(UserRecord)
_assistant_adapter = pydantic.TypeAdapter(AssistantRecord)
_summary_adapter = pydantic.TypeAdapter(SummaryRecord)
# Explicit annotations needed: these wrap union/Annotated types where mypy can't infer
_system_subtype_adapter: pydantic.TypeAdapter[SystemSubtypeRecord] = pydantic.TypeAdapter(SystemSubtypeRecord)
_system_fallback_adapter: pydantic.TypeAdapter[SystemRecord] = pydantic.TypeAdapter(SystemRecord)
_base_record_adapter: pydantic.TypeAdapter[BaseRecord] = pydantic.TypeAdapter(BaseRecord)
_file_history_adapter = pydantic.TypeAdapter(FileHistorySnapshotRecord)
_queue_operation_adapter = pydantic.TypeAdapter(QueueOperationRecord)
_custom_title_adapter = pydantic.TypeAdapter(CustomTitleRecord)
_progress_adapter = pydantic.TypeAdapter(ProgressRecord)
_pr_link_adapter = pydantic.TypeAdapter(PrLinkRecord)
_saved_hook_context_adapter = pydantic.TypeAdapter(SavedHookContextRecord)
_agent_name_adapter = pydantic.TypeAdapter(AgentNameRecord)
_ai_title_adapter = pydantic.TypeAdapter(AiTitleRecord)
_last_prompt_adapter = pydantic.TypeAdapter(LastPromptRecord)
_worktree_state_adapter = pydantic.TypeAdapter(WorktreeStateRecord)
_permission_mode_adapter = pydantic.TypeAdapter(PermissionModeRecord)
_session_mode_adapter = pydantic.TypeAdapter(SessionModeRecord)
_bridge_session_adapter = pydantic.TypeAdapter(BridgeSessionRecord)
_tag_adapter = pydantic.TypeAdapter(TagRecord)
_agent_color_adapter = pydantic.TypeAdapter(AgentColorRecord)
_agent_setting_adapter = pydantic.TypeAdapter(AgentSettingRecord)
_isolation_latch_adapter = pydantic.TypeAdapter(IsolationLatchRecord)
_speculation_accept_adapter = pydantic.TypeAdapter(SpeculationAcceptRecord)
_content_replacement_adapter = pydantic.TypeAdapter(ContentReplacementRecord)
_fork_context_ref_adapter = pydantic.TypeAdapter(ForkContextRefRecord)
_attachment_adapter = pydantic.TypeAdapter(AttachmentRecord)


def validate_session_record(
    data: Mapping[str, Any],
) -> SessionRecord:  # strict_typing_linter.py: loose-typing — raw JSON-parsed session record
    """Validate a session record dict using type-dispatch for performance.

    Dispatches to per-type TypeAdapters based on the 'type' field, avoiding
    the full SessionRecord left-to-right union scan. Branches are ordered by
    frequency (assistant 33%, queue-operation 27%, user 22%, progress 17%).

    For 'system' records, uses SystemSubtypeRecord (discriminator='subtype')
    first, falling back to generic SystemRecord if subtype is unknown.

    Falls back to SessionRecordAdapter for completely unknown types,
    preserving forward compatibility.

    This is the recommended entry point for validating session records.
    SessionRecordAdapter.validate_python() is equivalent but slower.
    """
    record_type = data.get('type')

    if record_type == 'assistant':
        return _assistant_adapter.validate_python(data)
    elif record_type == 'queue-operation':
        return _queue_operation_adapter.validate_python(data)
    elif record_type == 'user':
        return _user_adapter.validate_python(data)
    elif record_type == 'progress':
        return _progress_adapter.validate_python(data)
    elif record_type == 'system':
        if 'subtype' in data:
            try:
                return _system_subtype_adapter.validate_python(data)
            except pydantic.ValidationError:
                # Unknown subtype (e.g., new Claude Code feature) — use base record.
                # In strict mode this re-raises; in lenient mode BaseRecord accepts extras.
                return _base_record_adapter.validate_python(data)
        else:
            # Standard system record (no subtype) — use generic SystemRecord
            return _system_fallback_adapter.validate_python(data)
    elif record_type == 'summary':
        return _summary_adapter.validate_python(data)
    elif record_type == 'custom-title':
        return _custom_title_adapter.validate_python(data)
    elif record_type == 'file-history-snapshot':
        return _file_history_adapter.validate_python(data)
    elif record_type == 'pr-link':
        return _pr_link_adapter.validate_python(data)
    elif record_type == 'saved_hook_context':
        return _saved_hook_context_adapter.validate_python(data)
    elif record_type == 'agent-name':
        return _agent_name_adapter.validate_python(data)
    elif record_type == 'ai-title':
        return _ai_title_adapter.validate_python(data)
    elif record_type == 'last-prompt':
        return _last_prompt_adapter.validate_python(data)
    elif record_type == 'worktree-state':
        return _worktree_state_adapter.validate_python(data)
    elif record_type == 'permission-mode':
        return _permission_mode_adapter.validate_python(data)
    elif record_type == 'mode':
        return _session_mode_adapter.validate_python(data)
    elif record_type == 'bridge-session':
        return _bridge_session_adapter.validate_python(data)
    elif record_type == 'tag':
        return _tag_adapter.validate_python(data)
    elif record_type == 'agent-color':
        return _agent_color_adapter.validate_python(data)
    elif record_type == 'agent-setting':
        return _agent_setting_adapter.validate_python(data)
    elif record_type == 'isolation-latch':
        return _isolation_latch_adapter.validate_python(data)
    elif record_type == 'speculation-accept':
        return _speculation_accept_adapter.validate_python(data)
    elif record_type == 'content-replacement':
        return _content_replacement_adapter.validate_python(data)
    elif record_type == 'fork-context-ref':
        return _fork_context_ref_adapter.validate_python(data)
    elif record_type == 'attachment':
        return _attachment_adapter.validate_python(data)
    else:
        return SessionRecordAdapter.validate_python(data)


_workflow_journal_adapter: pydantic.TypeAdapter[WorkflowJournalRecord] = pydantic.TypeAdapter(WorkflowJournalRecord)


def validate_journal_record(
    data: Mapping[str, Any],  # strict_typing_linter.py: loose-typing — raw JSON-parsed journal record
) -> WorkflowJournalRecord:
    """Validate a Workflow run-journal entry (started/result).

    The journal is a distinct artifact from the session transcript, so its records
    are validated here rather than through the SessionRecord union.
    """
    return _workflow_journal_adapter.validate_python(data)
