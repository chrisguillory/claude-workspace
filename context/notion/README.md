# Notion Export Archive

A machine-local mirror of a Notion workspace — captured once from a full export so its content is semantically searchable as context. Gitignored and never leaves the machine; only this README is tracked.

> [!WARNING]
> This repo is **public**. Keep this file workspace-agnostic — never name a workspace, teamspace, project, page, or internal system. The archived data is machine-local (gitignored); this prose is not.

Notion nests workspace → teamspace → page → subpage; the export preserves that on disk. A page's name carries its 32-hex Notion id; a page with children becomes a folder. Teamspaces are top-level folders, so search can be scoped by path.

```text
context/notion/{workspace}/                       # {workspace} = the workspace's slug
├── Engineering/                                  # a teamspace
│   ├── Onboarding 1f2e3d4c….md                   # a leaf page → "{title} {page-id}.md"
│   └── Service Architecture 5a6b7c8d…/           # a page with subpages → a folder
│       ├── Ingestion 9e0f1a2b….md                # subpages nest arbitrarily deep,
│       └── Storage 3c4d5e6f….md                  #   and there can be many
├── Product/
│   └── Roadmap 7a8b9c0d….md
└── index.json                                    # page-id → title · path · notion.so URL
```

Built by `scripts/process-notion-export.py` (mechanics live in its docstring). Text is indexed into the `document-chunks` collection for search; binaries (images, video, data dumps) stay on disk as on-demand assets.
