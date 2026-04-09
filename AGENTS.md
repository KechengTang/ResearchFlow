# ResearchFlow Agent Guide

ResearchFlow is an agent-ready research knowledge base, not just a paper folder.

Core loop:

```text
collect paper list -> paper analysis -> build index -> research assist
```

`Download` sits inside the intake path between collection and analysis.

## What This Repository Is For

- Build a local literature knowledge base from web pages, GitHub awesome lists, and PDFs
- Convert papers into structured notes that can be queried by agents
- Reuse the knowledge base for comparison, idea generation, question banks, reviewer-style critique, and code-grounded implementation planning

## Source of Truth

- `paperCollection/`: first retrieval layer by task, technique, and venue
- `paperAnalysis/`: authoritative structured evidence
- `paperPDFs/`: local PDF storage
- `QuestionBank/`: question and challenge outputs
- `paperIDEAs/`: idea outputs
- `scripts/`: maintenance and automation helpers

## Working Rules

- For broad research questions, start from `paperCollection/`, then open relevant files in `paperAnalysis/`
- When analysis notes are added or updated, rebuild `paperCollection/` before answering KB-wide questions
- Prefer answers grounded in local note structure such as tags, venue, year, `core_operator`, and `primary_logic`
- Treat this repository as shared memory that can support Claude Code, Codex CLI, and other agents

## Skill Routing

- Reusable workflows live in `.agents/skills` and `.claude/skills`
- Use `.claude/skills/User_README.md` for the English quick skill map or `.claude/skills/User_README_ZN.md` for the Chinese version
- Main workflow families:
  - paper collection
  - PDF download and repair
  - paper analysis
  - collection index rebuild
  - knowledge-base query and comparison
  - question-bank and idea generation
  - reviewer-style stress testing
  - code-context paper retrieval

## Typical Agent Tasks

- Build or refresh a topic-specific knowledge base
- Compare multiple papers and extract transferable operators or design patterns
- Generate question banks and idea notes grounded in the current literature
- In a linked code repository, retrieve relevant papers before editing model- or method-related code

## Current User Research Tracks

The original repository contains motion-generation examples, but the current user's active research direction is Gaussian Splatting, especially:

- `3DGS_Editing`
- `3DGS_Reconstruction`
- `4DGS_Editing`
- `4DGS_Reconstruction`
- `Gaussian_Splatting_Foundation`

When adding new papers for the current user:

- Prefer the category folders above under both `paperPDFs/` and `paperAnalysis/`
- Keep using the shared `paperAnalysis/analysis_log.csv` as the intake log
- Assign `analysis_log.csv.importance` automatically for new Gaussian-track papers instead of leaving it blank; use the rubric in `GAUSSIAN_SPLATTING_KB_GUIDE.md` and bias slightly high because the user usually curates strong papers
- Treat the motion-related entries as reference examples unless the user explicitly asks about that domain
- For KB-wide answers, ground Gaussian Splatting discussions in the user's own categories first, then reuse the existing framework for compare / question-bank / idea generation
