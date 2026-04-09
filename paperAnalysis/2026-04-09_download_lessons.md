---
title: Download Lessons
date: 2026-04-09
scope: gaussian_splatting_intake
---

# Download Lessons From The 2026-04-09 Gaussian Splatting Intake

This note records a few practical lessons from the first large 3DGS/4DGS batch download.

## What Worked

- Small-batch downloading was much more reliable than one large batch.
- `curl.exe -L --fail --retry 2` worked better than large PowerShell `Invoke-WebRequest` jobs for long PDF pulls.
- Verifying the local file header with `%PDF` was a good final check after download.

## What Went Wrong

- Several papers were first filed by arXiv year even though their formal publication venue was different.
- Large batched downloads could time out midway, leaving the log and filesystem temporarily out of sync.
- Project pages and official proceedings pages sometimes exposed newer publication metadata than the original arXiv entry.

## Recommended Intake Rule

For future Gaussian Splatting papers:

1. Download the PDF first and confirm the file is valid.
2. Verify the formal publication venue before finalizing `venue`, `year`, and local path.
3. Prefer official proceedings PDFs when available.
4. If only arXiv PDF is available but a later venue is confirmed, keep the arXiv PDF link and store the paper under the confirmed venue-year path.
5. After bulk download, run one pass to reconcile `analysis_log.csv` with actual local files.

## Examples From This Batch

- `GaussianEditor`:
  - initial intake: `arXiv 2023`
  - corrected to: `CVPR 2024`
  - action: moved PDF to `paperPDFs/3DGS_Editing/CVPR_2024/` and updated the paper link to the official CVPR PDF

- `Dynamic-eDiTor`:
  - initial intake: `arXiv 2025`
  - corrected to: `CVPR 2026`
  - action: kept the arXiv PDF, but moved it to `paperPDFs/4DGS_Editing/CVPR_2026/`

- `4DGS-Craft`:
  - initial suspicion: possible later journal publication
  - current conservative record: `arXiv 2025`
  - action: kept the arXiv venue because accessible sources were not strong enough to safely promote it to a journal record

## Extra Rule Added

- Do not promote a paper from arXiv to a journal/conference venue unless the evidence is clearly accessible from an official or otherwise strong bibliographic source.
