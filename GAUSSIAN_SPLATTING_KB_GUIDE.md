---
created: 2026-04-09
updated: 2026-04-09
---

# Gaussian Splatting KB Guide

This note defines the personal ResearchFlow setup for Gaussian Splatting research.

## Scope

Primary research tracks:

- `3DGS_Editing`
- `3DGS_Reconstruction`
- `4DGS_Editing`
- `4DGS_Reconstruction`
- `Gaussian_Splatting_Foundation`

Optional side tracks if the literature grows:

- `Dynamic_Scene_Representation`
- `Text_Guided_3D_Editing`
- `View_Consistent_Generative_Editing`

## Folder Convention

Every new paper should follow the same intake path:

```text
external PDF -> paperPDFs/<Category>/<Venue_Year>/<Year>_<Title>.pdf
            -> analysis_log.csv state = Downloaded
            -> paperAnalysis/<Category>/<Venue_Year>/<Year>_<Title>.md
            -> analysis_log.csv state = checked
            -> rebuild paperCollection/
```

## Recommended Category Mapping

- Papers mainly about editing static Gaussian scenes -> `3DGS_Editing`
- Papers mainly about static scene reconstruction / optimization / representation -> `3DGS_Reconstruction`
- Papers mainly about dynamic Gaussian editing / controllable dynamic rendering -> `4DGS_Editing`
- Papers mainly about dynamic scene reconstruction / dynamic Gaussian representation -> `4DGS_Reconstruction`
- Foundational splatting papers, rendering, rasterization, deformation backbones, or general Gaussian representations -> `Gaussian_Splatting_Foundation`

## Recommended Technique Tags

Use the category as the first semantic anchor, then add 3 to 8 reusable tags such as:

- `gaussian-splatting`
- `3dgs`
- `4dgs`
- `scene-editing`
- `text-guided-editing`
- `semantic-editing`
- `geometry-preservation`
- `appearance-editing`
- `dynamic-scene`
- `deformation`
- `view-consistency`
- `optimization-based`
- `feed-forward`
- `interactive-editing`

## Importance Rating

For the current user's Gaussian Splatting tracks, `paperAnalysis/analysis_log.csv` should not leave `importance` blank once a paper is accepted into the working set.

Use an intentionally optimistic prior because these papers are usually hand-picked by the user:

- `S`: Directly supports the current core research line or appears as a key backbone, benchmark, teacher, or semantic grounding component in active idea notes such as `paperIDEAs/2026-04-09_4dgs_semantic_feedforward_ideas.md`.
- `A`: Clearly relevant and worth reading carefully; useful baseline, neighboring operator, or important comparison paper, but not one of the smallest core set.
- `B`: Good paper and still relevant to Gaussian Splatting, but more peripheral to the user's current build direction or mostly kept for coverage.
- `C`: Archive/reference only; keep in the KB, but low current priority.

Practical rules:

- Default to `A` rather than `B` when a new Gaussian paper is clearly in-scope but its exact role is not settled yet.
- Upgrade to `S` when the paper directly unlocks the user's current idea framing, factorization, semantic grounding, motion scaffold, or main baseline comparisons.
- Downgrade to `B` only when the paper is useful but not central to the active idea note or current experiment plan.
- When the user explicitly says a subtopic is highly important, bias one level upward if the paper is on that subtopic.

## Daily Workflow

### 1. Intake

- Put new PDFs into `paperPDFs/` with standardized names
- Append a row to `paperAnalysis/analysis_log.csv`
- Assign `importance` during intake using the rubric above; do not leave it blank for Gaussian-track papers
- Keep the paper in `Downloaded` until a real analysis note exists

### 2. Analysis

Analyze in small batches, ideally 1 to 4 papers per session:

- What problem is the paper really solving?
- What is the core operator?
- What constraints or priors make it work?
- What kind of edit / reconstruction control does it enable?
- What breaks under dynamic scenes, sparse views, or large edits?

### 3. Build

After new analysis notes are finished:

- rebuild `paperCollection/`
- browse by task first
- then jump into the analysis note for evidence

### 4. Research Assist

Use the KB for four outputs:

- compare tables across papers
- question banks for unresolved bottlenecks
- idea notes grounded in existing operators
- code-planning support before editing a reconstruction or rendering repo

## First Seed Paper

- `Variation-aware Flexible 3D Gaussian Editing`
  - category: `3DGS_Editing`
  - venue: `ICLR 2026`
  - status: `Downloaded`

## Next Step For Each Newly Added PDF

1. Read the abstract, intro, method, and core experiment sections.
2. Write a structured note into `paperAnalysis/`.
3. Assign or revise `importance` based on the current active idea note and the rubric above.
4. Add tags that are reusable across future Gaussian Splatting papers.
5. Mark the log row from `Downloaded` to `checked`.
6. Rebuild `paperCollection/`.

## Suggested Weekly Cadence

- 1 intake session: add 5 to 15 candidate papers
- 2 analysis sessions: finish 2 to 6 strong papers
- 1 comparison session: summarize one subtopic
- 1 idea session: convert gaps into `QuestionBank/` or `paperIDEAs/`
