# papers-sync-from-zotero

## Overview

Syncs papers from a Zotero library into ResearchFlow's structured layout. Two modes:

- **Mode A (API, recommended)**: Uses `pyzotero` + Zotero Web API for complete metadata, incremental sync, and identity tracking
- **Mode B (folder, fallback)**: Reads a flat PDF folder with no Zotero dependency; metadata extracted from PDF text

## Quick start

```
1. "Sync my Zotero library into ResearchFlow. Library ID is 12345, API key is xxx."
2. "Sync only the 'Video Generation' collection from Zotero."
3. "Import PDFs from ~/Downloads/papers/ — they're all CVPR 2025 papers."
4. "Re-sync Zotero — only pull papers added since last sync."
```

## What it does

1. Connects to Zotero (API) or scans a PDF folder (fallback)
2. Fetches/extracts metadata (title, authors, venue, year, DOI, tags)
3. Copies (never moves) PDFs into `paperPDFs/<Category>/<Venue_Year>/`
4. Writes rich metadata to `paperAnalysis/processing/zotero/manifest.jsonl`
5. Appends lightweight rows to `paperAnalysis/analysis_log.csv`

## What it does NOT do

- Does not analyze PDFs (use `papers-analyze-pdf` after sync)
- Does not sync annotations (Phase 2)
- Does not modify or move Zotero originals
- Does not access Zotero SQLite directly (API only)

## Output files

| File | Content |
|------|---------|
| `paperPDFs/<Category>/<Venue_Year>/<Year>_<Title>.pdf` | Copied PDFs in RF naming convention |
| `paperAnalysis/processing/zotero/manifest.jsonl` | Rich metadata (all Zotero fields) |
| `paperAnalysis/processing/zotero/sync_state.json` | Incremental sync state |
| `paperAnalysis/processing/zotero/category_map.json` | Zotero collection/tag -> RF category mapping |
| `paperAnalysis/analysis_log.csv` | Lightweight rows (state=Downloaded) |

## After sync

Run `papers-analyze-pdf` to process the newly synced `Downloaded` entries, then `papers-build-collection-index` to update indexes.
