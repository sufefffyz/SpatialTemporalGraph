---
name: log-progress
description: Record and sync current project progress to Obsidian project notes. Use when the user asks to record project progress, sync Obsidian notes, update progress.md, decisions.md, or backlog.md, maintain project notes, or summarize the current session into append-only project logs.
---

# Log Progress

Sync the current session into the project's Obsidian notes without changing unrelated code.

## Workflow

1. Read the repository `AGENTS.md` first.
2. Confirm the target Obsidian project directory and note file names from `AGENTS.md`. If no override is given, default to:
   - `Projects/<project-name>/progress.md`
   - `Projects/<project-name>/decisions.md`
   - `Projects/<project-name>/backlog.md`
3. Read the current contents of all three note files before editing them. If a file is missing, create it with the bundled script.
4. Extract only facts that are confirmed by the current session, repo state, or explicit user instruction.
5. Write all note content in Chinese.
6. Preserve the exact progress section names `Done`, `Blockers`, and `Next`, but keep the bullet content in Chinese.
7. Append only. Do not delete, rewrite, or "clean up" prior history unless the user explicitly asks.
8. If the request is only to record progress, do not modify business code.

## What To Record

- `progress.md`
  - Append a new entry under today's date.
  - Include exactly these sections:
    - `Done`
    - `Blockers`
    - `Next`
- `decisions.md`
  - Append only when the current session produced a stable conclusion, design choice, or workflow rule.
- `backlog.md`
  - Append unchecked checklist items only for real follow-up work discovered in this session.

## Bundled Script

Use [`scripts/update_obsidian_notes.py`](scripts/update_obsidian_notes.py) for deterministic note updates after you have gathered the facts.

Typical usage:

```bash
python3 .agents/skills/log-progress/scripts/update_obsidian_notes.py \
  --repo-root /absolute/path/to/repo \
  --vault "/absolute/path/to/Obsidian Vault" \
  --project SpatialTemporalGraph \
  --done "已完成的数据集处理逻辑梳理" \
  --blocker "无" \
  --next "运行相关测试并检查结果" \
  --decision "默认将项目进展同步到 Obsidian 项目目录" \
  --backlog "补充缺失的回归测试"
```

Script behavior:

- Create missing note files with minimal frontmatter and headings.
- Append progress entries under today's date with a time-stamped session block.
- Append dated bullets to `decisions.md`.
- Append unchecked checklist items to `backlog.md`.

## Guardrails

- Do not invent work that was not actually completed.
- If information is incomplete, record only what can be confirmed and leave the rest out.
- Keep entries concise, factual, and easy to scan.
- If the Obsidian vault is outside the writable workspace, request approval before writing.
- If existing templates still contain placeholders such as `<PROJECT_NAME>`, do not rewrite old content unless the user asks; append new confirmed entries using the real project name.
