---
name: papers-sync-from-zotero
description: "Syncs papers from a Zotero library into RF's structured layout via pyzotero / Zotero API (primary) or flat PDF folder import (fallback). Copies PDFs into `paperPDFs/<Category>/<Venue_Year>/`, writes rich metadata to `paperAnalysis/processing/zotero/manifest.jsonl`, and appends lightweight rows to `analysis_log.csv`. Supports incremental sync. Use when the user wants to import papers from Zotero into the ResearchFlow pipeline."
---

# Papers Sync From Zotero

## Purpose

Bridge Zotero (bibliographic source of truth) into ResearchFlow (analytical source of truth) by syncing papers, PDFs, and metadata into RF's structured layout.

ResearchFlow should not replicate Zotero. Zotero handles collection, metadata cleaning, PDF/annotation, and citation styles. RF handles structured analysis, comparison, brainstorming, and agent retrieval. This skill connects the two.

## Pipeline position

```
Zotero library
  -> papers-sync-from-zotero (this skill)
    -> paperPDFs/<Category>/<Venue_Year>/<Year>_<Title>.pdf
    -> paperAnalysis/processing/zotero/manifest.jsonl (rich metadata)
    -> paperAnalysis/analysis_log.csv (lightweight rows)
  -> papers-analyze-pdf
  -> papers-build-collection-index
  -> papers-query-knowledge-base
```

## Two operating modes

### Mode A: Zotero API (primary, recommended)

Uses `pyzotero` library to connect to Zotero via Web API or local API.

Advantages:
- Complete metadata (title, authors, venue, DOI, tags, collections, dates, abstract)
- Incremental sync via `since` / `version` parameters
- Identity tracking via `zotero_key` + `zotero_library_id`
- No guessing from PDF text

Requirements:
- `pyzotero` installed (`pip install pyzotero`)
- Zotero API key (for Web API) or Zotero running locally (for local API)
- User provides: library ID + API key, or confirms local Zotero is running

### Mode B: Flat PDF folder (fallback)

Reads a local folder of PDFs with no Zotero dependency.

Advantages:
- Works without Zotero installed
- Works with any PDF collection (not just Zotero exports)

Limitations:
- Metadata extraction from PDF text is unreliable
- No incremental sync (full scan + dedup each time)
- No identity tracking

Use when: user has no Zotero, or has a one-off folder of PDFs to import.

## Input

User provides:

**Mode A (API):**
- **Library type**: `user` or `group`
- **Library ID**: Zotero library ID (numeric)
- **API key**: Zotero API key (for Web API) — or confirm local Zotero is running
- **Collection filter** (optional): sync only a specific Zotero collection
- **Category hint** (optional): override RF category for all items in this sync
- **Tag filter** (optional): sync only items with specific Zotero tags

> **Security recommendation:** Prefer Mode B (folder + BibTeX) over Mode A when possible. If Mode A is used, the user should create a **read-only** API key in Zotero settings (Settings → Feeds/API → Create new private key → uncheck "Allow write access"). Never store the API key in committed files. The agent should not echo the key in output or logs.

**Mode B (folder):**
- **PDF folder path**: e.g. `/home/user/Zotero/storage` or any flat folder
- **Category hint** (optional): e.g. `Motion_Generation_Text_Speech_Music_Driven`
- **Venue/year hint** (optional): e.g. `CVPR 2025`
- **BibTeX / CSV export** (optional): for more accurate metadata

## Output

### 1. PDFs in RF structure

```
paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf
```

Naming convention:
- SanitizedTitle: words joined by `_`, no spaces/commas/hyphens/colons
- Example: `2025_MotionDiffuse_Text_Driven_Human_Motion_Generation.pdf`
- Venue_Year: `CVPR_2025`, `ICLR_2026`, `arXiv_2025`, `unknown_2026`

### 2. Rich metadata manifest

```
paperAnalysis/processing/zotero/manifest.jsonl
```

One JSON object per line, containing all Zotero metadata:

```json
{
  "zotero_key": "ABC12345",
  "zotero_library_id": "12345",
  "zotero_version": 42,
  "title": "MotionDiffuse: Text-Driven Human Motion Generation",
  "authors": ["Zhang, M.", "Cai, Z.", "Pan, L."],
  "year": 2025,
  "venue": "CVPR",
  "doi": "10.1109/CVPR.2025.xxxxx",
  "abstract": "We present MotionDiffuse...",
  "citekey": "zhang2025motiondiffuse",
  "zotero_tags": ["motion-generation", "diffusion"],
  "zotero_collections": ["Motion Generation"],
  "pdf_source_path": "/home/user/Zotero/storage/ABC12345/Zhang_2025.pdf",
  "rf_pdf_path": "paperPDFs/Motion_Generation/CVPR_2025/2025_MotionDiffuse_Text_Driven_Human_Motion_Generation.pdf",
  "rf_category": "Motion_Generation_Text_Speech_Music_Driven",
  "rf_sort": "Motion_Generation",
  "synced_at": "2026-04-13T18:00:00",
  "sync_version": 1
}
```

### 3. Lightweight rows in analysis_log.csv

```
state,importance,paper_title,venue,project_link_or_github_link,paper_link,sort,pdf_path
```

| Column | Value |
|--------|-------|
| state | `Downloaded` |
| importance | (empty) |
| paper_title | from Zotero metadata |
| venue | `<Venue> <Year>` (e.g. `CVPR 2025`) |
| project_link_or_github_link | (empty — fill later) |
| paper_link | DOI link or arXiv link if available |
| sort | derived from RF category |
| pdf_path | RF-convention path |

## Workflow

### Mode A: Zotero API sync

**Step 1: Connect and authenticate**
- Check if `pyzotero` is installed; if not, prompt user to install
- Connect to Zotero library using provided credentials
- Verify connection by fetching library version

**Step 2: Fetch items (incremental)**
- Read last sync version from `paperAnalysis/processing/zotero/sync_state.json`:
  ```json
  {"library_id": "12345", "last_version": 42, "last_synced": "2026-04-13T18:00:00"}
  ```
- If first sync: fetch all items
- If incremental: fetch only items with `version > last_version` using `since` parameter
- Apply collection/tag filters if specified
- Filter to item types with PDF attachments

**Step 3: Map metadata to RF structure**
- For each Zotero item:
  - Map `itemType` + `publicationTitle` / `conferenceName` / `proceedingsTitle` → RF `venue`
  - Map Zotero `tags` + `collections` → RF `category` (using a configurable mapping table)
  - If category cannot be determined: mark as `Uncategorized` and flag for user input
- Present mapping table to user for confirmation (especially first sync):
  ```
  | # | Zotero title | Year | Venue | -> RF Category | -> RF Venue_Year | Status |
  ```

**Step 4: Download/copy PDFs**
- For each confirmed item:
  - Locate PDF attachment via Zotero API (`/items/{key}/children`)
  - If using Web API: download PDF to RF path
  - If using local API / local storage: copy (never move) PDF to RF path
  - Sanitize filename following RF convention
  - Skip if target already exists

**Step 5: Write manifest + log**
- Append rich metadata to `paperAnalysis/processing/zotero/manifest.jsonl`
- Append lightweight row to `paperAnalysis/analysis_log.csv`
- Update `sync_state.json` with new version number

### Mode B: Flat folder import

**Step 1: Link source folder**
- If outside RF workspace: `ln -s /path/to/pdfs linkedCodebases/zotero-pdfs`
- If accessible: read directly

**Step 2: Scan and dedup**
- Walk folder recursively, list all `.pdf` files
- Fuzzy match against existing `analysis_log.csv` `paper_title` column (similarity > 0.8 = skip)

**Step 3: Extract metadata**
Priority order:
1. BibTeX / CSV export (if provided) — match by filename or title
2. PDF metadata fields (`/Title`, `/Author`, `/CreationDate`)
3. PDF first-page text (title, venue from header/footer)
4. User hints + filename parsing (fallback)

Present results table for confirmation:
```
| # | Original filename | Detected title | Year | Venue | Category | Status |
```

**Step 4: Copy and organize**
- Copy (never move) to `paperPDFs/<Category>/<Venue_Year>/<Year>_<SanitizedTitle>.pdf`
- Skip existing targets

**Step 5: Write manifest + log**
- Same as Mode A, but `zotero_key` and API fields are empty
- `pdf_source_path` records original location

## Category mapping

For Mode A, Zotero collections/tags are mapped to RF categories. Default mapping can be configured in `paperAnalysis/processing/zotero/category_map.json`:

```json
{
  "zotero_collection_to_rf_category": {
    "Motion Generation": "Motion_Generation_Text_Speech_Music_Driven",
    "Video Generation": "Video_Generation",
    "3D Gaussian Splatting": "3D_Gaussian_Splatting"
  },
  "zotero_tag_to_rf_category": {
    "motion-gen": "Motion_Generation_Text_Speech_Music_Driven",
    "video-diffusion": "Video_Generation"
  },
  "default_category": "Uncategorized"
}
```

If no mapping matches, agent infers from title/abstract keywords or asks user.

## Optional frontmatter enrichment

After sync, the following optional fields become available for `papers-analyze-pdf` to include in analysis note frontmatter:

| Field | Source | Required? |
|-------|--------|-----------|
| authors | Zotero API | optional |
| doi | Zotero API | optional |
| citekey | Better BibTeX / Zotero | optional |
| zotero_key | Zotero API | optional |
| abstract | Zotero API | optional |

These fields are stored in `manifest.jsonl` and can be pulled into frontmatter by `papers-analyze-pdf` if present. Existing analysis notes without these fields remain valid.

## Constraints

- **Copy, never move** source PDFs (Zotero library stays intact)
- **Incremental by default** (Mode A) — only sync new/updated items
- **No Zotero database direct access** — use API only (Zotero docs explicitly warn against direct SQLite access)
- **No annotation sync in this skill** — that is a separate Phase 2 concern
- **No PDF analysis** — that is `papers-analyze-pdf`
- **Idempotent** — re-running skips already-synced items (by `zotero_key` or fuzzy title match)
- **analysis_log.csv stays lightweight** — rich metadata lives in manifest.jsonl

## Relationship to other skills

| Skill | Role | Boundary |
|-------|------|----------|
| `papers-sync-from-zotero` (this) | Source adapter: Zotero -> RF structure | Stops after PDF copy + manifest + log append |
| `papers-collect-from-github-awesome` | Source adapter: GitHub repos -> analysis_log.csv | Collects metadata only, no PDFs |
| `papers-collect-from-web` | Source adapter: web pages -> analysis_log.csv | Collects metadata only, no PDFs |
| `papers-download-from-list` | Downloads PDFs from URLs in triage lists | Handles online sources, not local files |
| `papers-analyze-pdf` | Analyzes PDFs into structured Markdown notes | Assumes PDFs already at RF paths |

## Typical Usage

> "Sync my Zotero library into ResearchFlow. My library ID is 12345 and API key is xxx."

> "Sync only the 'Video Generation' collection from Zotero."

> "Import PDFs from ~/Zotero/storage into ResearchFlow, auto-detect categories."

> "I have a BibTeX export at ~/refs.bib and PDFs at ~/papers/. Use the BibTeX for metadata."

> "Re-sync Zotero — only pull papers added since last sync."
