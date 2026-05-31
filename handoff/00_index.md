# DriveSort — Handoff Index

This folder documents the project for continuation in Claude Code.
Read these files in order on first pickup; jump to specific ones when working on a feature.

## Files in this folder

| File | When to read it |
|---|---|
| `01_project_overview.md` | Always — what this is, what it does, current status |
| `02_architecture.md` | Before touching any module — component map and data flow |
| `03_design_decisions.md` | Before changing anything significant — the *why* behind choices |
| `04_codebase_reference.md` | Working on a specific module — precise API contracts |
| `05_known_issues_and_next_steps.md` | Starting a new session — what's incomplete and what to build next |
| `06_drive_context.md` | Working on classification logic — actual Drive structure this was designed for |

## Quick state summary

- **Status:** Code complete, not yet run against real Drive
- **Bootstrap:** Not yet executed — no `data/taxonomy.json` exists yet
- **Tests:** Scaffold only (`tests/__init__.py`) — no test code written
- **Immediate next task:** Write tests for `taxonomy.py`, then run bootstrap
- **Biggest known gap:** When a new category is created from a single novel file in `scanner.py`, its taxonomy entry is never actually added (the `add_category` call is missing after the folder creation)

## Running the project

```bash
# From the drivesort/ project root:
python -m venv .venv && source .venv/bin/activate
pip install -e .
ollama pull phi3:mini
# Place credentials.json at data/credentials.json
drivesort bootstrap
drivesort scan --live
```
