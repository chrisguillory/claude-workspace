# preflight-check

Machine and mesh readiness checks for Claude Code sessions — *is this machine in a good state to build right now?* Each check detects one operational problem, explains it, and points at an actionable fix.

**Standalone-first:** run it yourself, no Claude required.

## Usage

```bash
preflight-check check                    # run all checks on the current machine
preflight-check fix dns_resolver_wedge   # apply a check's remediation + verify (sudo)
```

Exit codes: `0` healthy, `1` warning, `2` critical.

## Install

```bash
uv tool install --editable mcp/preflight-check
claude mcp add --scope user preflight-check -- preflight-check-mcp   # optional: expose to Claude
```
