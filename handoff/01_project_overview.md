# 01 — Project Overview

## What DriveSort is

A local-AI system that organises Google Drive files into folders automatically.
It has two distinct modes:

**Bootstrap (run once):** Discovers what categories *should* exist by clustering
all your files using unsupervised ML. A human reviews each cluster, names or
rejects it, and the result becomes the taxonomy. No labels needed upfront.

**Ongoing scan (run periodically):** Classifies new/unorganised files against
the established taxonomy. High-confidence files are auto-moved. Uncertain files
go to a human review queue. Files that don't fit any category are flagged as
novel, accumulated, and periodically re-clustered to detect emerging categories.

## What makes it distinctive

**Open-set classification** — the system can say "this doesn't belong anywhere"
rather than being forced to pick a wrong folder. This is the core novelty.
Most Drive organisation tools assume a fixed, predefined set of categories.
This one learns what categories *your* Drive naturally contains, and evolves as
your content evolves.

**Everything runs locally** — sentence-transformers (~80 MB), HDBSCAN, UMAP,
and Ollama (phi3:mini, ~2 GB) run on your machine. No data leaves your network.
No API keys needed beyond the Google OAuth credential for Drive access.

## What it does NOT do

- Delete files (ever)
- Rename files
- Touch file content
- Read full file content (uses Drive API's content snippet, max ~400 chars)
- Move files without confirmation (dry-run is default)

## Current status

The full codebase was written in one session. It has never been run against a
real Drive. The bootstrap has not been executed, so no `data/taxonomy.json`
exists. The code is logically complete but has at least one known bug and no
tests. See `05_known_issues_and_next_steps.md`.

## Target Drive

This was designed around a specific personal Drive with the following structure.
The taxonomy will reflect this when bootstrap is run:

- Career & Professional (LinkedIn docs, CV, quote PDFs)
- Finance & Investments (share scenarios, pricing spreadsheets, mortgage)
- FPV & RC Hobbies (Rotorflight, EdgeTX, ELRS, chimera.hex, wiring diagrams)
- Tech & Projects (3D printer firmware, project docs)
- Pets & Family (labradoodle pet insurance, family certificates)
- Media & Photos (IMG_*.png, screenshots)
- Property File (house documents, pool inspection)
- Archive (Takeout, Backup, Books, old data)
- XJR400R (Yamaha motorcycle service manuals, wiring)
- Car (vehicle documents)
- Snowrunner (gaming shortcuts and spreadsheets)
- Carrier Remote (HVAC IR decoding project: CSV data, timing spreadsheets)

See `06_drive_context.md` for the actual file inventory.
