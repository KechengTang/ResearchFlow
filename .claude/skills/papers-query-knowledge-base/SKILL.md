---
name: papers-query-knowledge-base
follows: rf-obsidian-markdown
description: Queries the local `paperAnalysis/` notes for research. Finds papers by title, task path, technique tags, venue, or year; summarizes methods and evidence across papers; cites core_operator and primary_logic from analysis frontmatter. When present, `paperCollection/index.jsonl` serves as the fast filter layer and `paperCollection/` pages serve navigation and backlink exploration.
---

# Paper Knowledge Base (paperAnalysis-first)

Use this skill to query the local paper corpus across conversations and projects. The main evidence layer is **paperAnalysis** (analysis notes with TL;DR, Part I/II/III, PDF links). When it exists, **paperCollection/index.jsonl** is the fast filter layer for large vaults, while the Markdown pages in **paperCollection/** remain useful for Obsidian jumps and backlink-friendly browsing.

**Where the knowledge base lives:** Under the current repository root that contains `paperAnalysis/` and `paperPDFs/`. `paperCollection/` is a generated companion layer created by `papers-build-collection-index`.

## Paths

- Use repo-relative paths rooted at the folder that contains `paperAnalysis/` and `paperPDFs/`.
- When invoking from another workspace, replace these with the correct absolute repository path for your machine.

## Statistics (current sample values)

| Dimension | Count |
|-----------|--------|
| Papers | 254 |
| Tasks | 7 |
| Technique tags | 587 |
| Venues | 17 |

Tasks: Human_Human_Interaction, Human_Object_Interaction, Human_Scene_Interaction, Motion_Controlled_ImageVideo_Generation, Motion_Editing, Motion_Generation_Text_Speech_Music_Driven, Motion_Stylization.

## Where things live

Relative to the repository root:

- **Analysis notes**: `paperAnalysis/<Category>/<Venue_Year>/<Year>_<Title>.md`; PDF path in frontmatter `pdf_ref`.
- **Generated index / navigation**: `paperCollection/index.jsonl`, `paperCollection/README.md`, `_AllPapers.md`, `by_task/`, `by_technique/`, `by_venue/`.

All paths use forward slashes.

## How to use for research

1. **Find papers** — Search `paperAnalysis/` directly by title, task folder, tags, venue, year, `core_operator`, or `primary_logic`. If `paperCollection/index.jsonl` is present, use it first to narrow a large candidate set, then read the matching notes for evidence.

2. **Read an analysis** — Open the matched analysis note directly. Each note has: **Frontmatter** (title, venue, year, category, tags, pdf_ref, core_operator, primary_logic); **TL;DR** (Summary, Key Performance); **Part I / II / III** sections whose exact headings follow `analysis_language` in `AGENTS.md`; and **Local Reading** (PDF). Older notes may still use legacy mixed headings, so match semantically rather than by one exact title string.

3. **Use `paperCollection/` when helpful** — `index.jsonl` is for fast filtering; the Markdown pages are for task / technique / venue overview pages, statistics, Obsidian jumps, or backlink exploration.

4. **Cite in answers** — Use **core_operator** and **primary_logic** from frontmatter for one-line method summary; **Summary** and **Key Performance** from TL;DR for comparison. Link to note: `[[paperAnalysis/.../file.md|Title]]`; to PDF: `[[paperPDFs/.../file.pdf|PDF]]`.

## Index frontmatter (for tooling)

Collection pages: `type: paper-index`, `dimension: task | technique | venue | all`, and optional `task:` / `technique:` / `venue:`.

For exact paths and frontmatter schema, see [references/structure.md](references/structure.md).

## Regenerating the collection (optional)

If you want refreshed `paperCollection/index.jsonl` and the Obsidian navigation pages, from the repository root:

```bash
python .claude/skills/papers-build-collection-index/scripts/build_paper_collection.py
```

Only notes with valid `pdf_ref` (path under `paperPDFs/` ending in `.pdf`) are indexed.

## Output style

- KB-grounded answers to research questions
- evidence-backed cross-paper summaries in prose
- direct pointers to relevant `paperAnalysis/...` notes and local PDFs when useful

## Boundaries

- This skill handles both text summaries and comparison requests. When comparing papers, provide honest text-based analysis that acknowledges metric incomparability (different datasets, evaluation protocols, etc.) rather than forcing misleading tabular comparisons.
- This skill can point out evidence-backed research gaps, but it does not generate systematic idea candidates, scope cuts, or experimental roadmaps. Use `research-brainstorm-from-kb` or `idea-focus-coach` for that.
- This skill does not issue reviewer-style acceptance or rejection risk verdicts. Use `reviewer-stress-test`.
- If the retrieval goal is to support an upcoming code change, prefer `code-context-paper-retrieval`.
