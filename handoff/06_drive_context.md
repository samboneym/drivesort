# 06 — Drive Context

The actual Google Drive this system was designed and tested against.
Useful for understanding why certain design choices were made, and for
writing realistic test fixtures.

Owner: samant.maharaj@gmail.com

---

## Root-level folder structure (as of May 2026)

### Organised (taxonomy already exists for these)
| Folder | Drive ID | Notes |
|---|---|---|
| Career & Professional | `13d_Eqqe82vTdBDRZFaAQfNQARiEO_pwu` | LinkedIn docs, CV, quote |
| Pets & Family | `1mzA3ttvdMuzbiouSOjP_fHHxAspH1Fwx` | Pet insurance comparison |
| Tech & Projects | `13BG-VCOpsSDEVZG0mTgD3EaEAlbUgYYH` | Marlin firmware, project guide |
| Finance & Investments | (subfolder of root, ID from tracker) | Share scenarios, pricing |
| FPV & RC Hobbies | `1EdoCq07ju3a8-AUxDFV3GgsWhl5qNunb` | Rotorflight, EdgeTX, wiring |
| Media & Photos | `1NyZXI35PCYnFh7_xGvS6ZRJXgUgUKag5` | IMG_1683–1688.png |
| Archive | `1KCVKISv4mKBWszqjlpdCcn-VytFgJOS8` | |
| Property File | `1APV6WfiAP0VuyQDIgb-Bc8C52f4zDM4A` | Pool inspection PDF |
| XJR400R | `1r0IMPZRPjfp2Ir3xACaGWlM4-4BizFEV` | Yamaha motorcycle |
| Car | `1Xw6KjVikzbZr6GTmauAAuICxJsj6e4HI` | Scraped JS files |
| Snowrunner | `1FyGMz5JpRFixBKl7I-sUK_qcAVvqktGM` | Gaming shortcuts |
| Opal | `1-nvsQIHbFEYwEE3t0qyi7KZr766ZDPqx` | Google AI notebook screenshots |
| Google AI Studio | `1YniaOASWriToaDiOBl8hvphFigEEV79-` | |

### Unorganised (candidates for moving during bootstrap)
| Folder | Drive ID | Suggested destination |
|---|---|---|
| Pictures | `0B6vBAJpDXKtYYXVBRmRienJUVnM` | Media & Photos |
| Documents | `0B6vBAJpDXKtYMllXdUdqSExfMVk` | Property File or rename |
| Books | `0B6vBAJpDXKtYZ0ptUDB4VG9rOVk` | Archive |
| Backup | `0B6vBAJpDXKtYVDNzZ0xCZGtxWlE` | Archive |
| Takeout | `0B6vBAJpDXKtYfjJUT0hNcG9hZEw5...` | Archive |
| Saved from Chrome | `1Gd4--3-49TD-utZCzhLa4uUYu7R3pxuD` | Archive |
| Financials | `0B6vBAJpDXKtYQVRkRks1cGN3X3M` | Finance & Investments |
| Carrier Remote | `174m4KRI30im-oSnCzHuRvGhoAq23_lcc` | Tech & Projects |
| HHGTTG | (not seen in API scan) | Archive |

---

## Notable individual files

### FPV & RC Hobbies cluster
These are the clearest natural cluster in the Drive. All in `1EdoCq07ju3a8-AUxDFV3GgsWhl5qNunb`:
- `chimera.hex` — Rotorflight firmware binary
- `FanController.zip` — flight controller project zip
- `EdgeTX-MT12-ELRS-2.10.5-Factory-SD-Content-2024-10-23.zip` — shortcut
- `Copy of Rotorflight-Target-Builder v2.0` — spreadsheet
- `Receiver VBat Calibration` — spreadsheet
- `FFT Sampling Calculator` — spreadsheet
- `R80 wiring diagram version 8` — Google Doc

### Career cluster
Multiple versions of LinkedIn doc (a known duplication problem):
- `LinkedIn_Optimisation_Samant_Maharaj_v6.docx` (latest, keep)
- `LinkedIn_Optimisation_Samant_Maharaj_v4.docx`
- `LinkedIn_Optimisation_Samant_Maharaj_v3.docx`
- 5× `LinkedIn_Optimisation_Samant_Maharaj.docx` (unnamed duplicates)
- `Samant Maharaj Quote.pdf`
- `discord_backup_codes.txt` — misplaced (Career & Professional parent)

### XJR400R cluster
Yamaha XJR400R motorcycle files:
- `XJR400_servicemanual.pdf` (shortcut)
- `YAMAHA_XJR_1200-1300_(HAYNES).pdf` (shortcut)
- `Electrical wiring diagram.txt`
- `wiring.pdf`
- `wiring.png`
- `Screenshot 2023-09-16 at 10.41.10 AM.png`
- `.DS_Store` — macOS metadata file (should be ignored/deleted, not classified)

### Carrier Remote cluster
HVAC IR remote decoding project, a "Tech & Projects" sub-project:
- `Carrier 32bit Decoding` — Google Sheets with IR timing data
- `Timing` — Google Sheets
- `Timing.ods` — ODS version
- `data.csv` — raw IR capture data (21 KB)

### Ambiguous files
- `Zombie-8384_manual.pdf` — likely FPV aircraft manual, in `Car`'s parent folder path
- `e296fc5962bc36c5168375b52bf16c7f.pdf` — baptism certificate in `Saved from Chrome`
- `190316-thermostatinstallers-v5web.pdf` — thermostat manual in `Documents/Product Manuals`
- `TP0218-telco-summary-EB-HA-Fibre 200200-CD_02.pdf` — ISP fibre summary, now historical
- `default_*.js` files in `Car` — scraped JS from a web page (car listing?), archive candidates

---

## Known classification edge cases for this Drive

**`.DS_Store`** — macOS metadata file. Should be filtered out before embedding,
not classified into any folder. It's currently in `XJR400R/`. Consider adding
a blocklist in `iter_files()`:

```python
SKIP_FILENAMES = {".DS_Store", "Thumbs.db", ".gitignore"}
SKIP_MIME_TYPES = {"application/vnd.google-apps.shortcut"}  # optionally
```

**Multiple LinkedIn versions** — they'll all cluster together correctly
(Career & Professional) but the duplication problem itself is out of scope
for DriveSort. DriveSort moves files; it doesn't delete duplicates.

**The `Car` folder with JS files** — a web scrape of a car listing page
accidentally saved. These JS files (`default_*.js`, ~10 files) are unlikely
to cluster with anything meaningful. They'll probably land as outliers or
weakly in Tech & Projects. Archive is the right destination.

**Shortcuts** (`application/vnd.google-apps.shortcut`) — the Drive contains
several: EdgeTX zip, Snowrunner spreadsheets, XJR400R service manual links.
The Drive API returns these but they have no content snippet. They'll embed
on filename only. This is usually sufficient but may cause misclassification
for ambiguously named shortcuts.

**Finance & Investments subfolder** (`1YvZL5kZta1MWsbRl2b8tVSiVlXEZLNdk`) —
this folder ID appears as `parentId` for `Scenario: 9000000 shares...` and
`Northern Rocks Pricing` but the folder itself wasn't returned in the root
scan results. It may be a subfolder inside another folder. Worth checking
the Drive structure manually before bootstrap.

---

## What the Drive Organisation Tracker says

A Google Sheets document (`Drive Organisation Tracker`,
`1kiyBgom6vQO_5oVgKuAuhZgLiK696tTKATZ4WJA0htQ`) already exists with a
manually curated list of files and their recommended destinations. This could
serve as ground-truth labelled data for evaluating the classifier's accuracy
after bootstrap. Compare DriveSort's classifications against this tracker
to measure how well the automated system matches human intent.
