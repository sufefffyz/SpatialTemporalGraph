# Project rules for Codex + Obsidian logging

When the user asks to record progress, sync project notes, update Obsidian notes, or mentions `progress.md`, `decisions.md`, or `backlog.md`, use the `log-progress` skill from `.agents/skills/log-progress/`.

Default Obsidian project directory:
`/Users/richardo/Nutstore Files/我的坚果云/Obsidian Vault/Projects/SpatialTemporalGraph/`

Default files:
- `progress.md`
- `decisions.md`
- `backlog.md`

## Required behavior

1. Read existing note content before editing.
2. Append only. Never delete or overwrite previous history unless the user explicitly asks.
3. Record only facts that are confirmed by the current session, repo state, or user instructions.
4. Update `progress.md` under today's date using exactly these sections:
   - `Done`
   - `Blockers`
   - `Next`
5. Write all recorded note content in Chinese, while preserving the section titles `Done`, `Blockers`, and `Next`.
6. If a stable conclusion, design choice, or workflow rule is confirmed, append it to `decisions.md`.
7. If new follow-up work is discovered, append it to `backlog.md` as an unchecked task.
8. Keep entries concise, factual, and easy to scan.
9. If the user only asks to record progress, do not modify business code.
10. If the user asks to both do work and record it, complete the work first, then update the notes.
11. If any note file is missing, create it in the default Obsidian project directory.

## Path reminder

Before first use, replace `<PROJECT_NAME>` with the actual project folder name in this file and in the Obsidian note templates.
