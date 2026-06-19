# llm-canvass

Runs a structured LLM judgment task across every in-scope file in the repo, in parallel.

## CLI

```
.claude/skills/llm-canvass/plan-slices.py --help
```

## Future considerations

- `--brief @path/to/brief.md` — supply a reusable brief file instead of generating one from the task argument.
- `--no-ignore` passthrough to `rg` — opt in to scanning vendored / generated / `.gitignore`'d code.
