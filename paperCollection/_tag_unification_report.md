---
type: tag-audit
tags:
  - paperCollection
  - maintenance/tags
generated: 2026-04-29T15:17
---

# Tag unification report (paperAnalysis → paperCollection)

This report identifies **synonymous or near-variant** tags in `paperAnalysis` as a basis for later unification.

## Summary

- raw unique tags: 541
- near-duplicate groups (variants>1): 0

## Top near-duplicate groups (by total frequency)


## Raw tag frequency (top 120)

- `status/analyzed`: 136
- `gaussian-splatting`: 132
- `4dgs`: 65
- `4DGS_Reconstruction`: 53
- `Gaussian_Splatting_Foundation`: 43
- `3dgs`: 37
- `3DGS_Editing`: 27
- `opensource/full`: 16
- `feed-forward`: 16
- `monocular-video`: 15
- `Motion_Generation_Text_Speech_Music_Driven`: 13
- `dynamic-scene`: 12
- `task/text-to-motion`: 11
- `dataset/humanml3d`: 11
- `diffusion`: 11
- `4DGS_Editing`: 9
- `scene-editing`: 9
- `vq-vae`: 8
- `autoregressive`: 8
- `multi-view-consistency`: 8
- `repr/smpl`: 7
- `temporal-consistency`: 7
- `stylization`: 7
- `repr/humanml3d-263d`: 6
- `deformable-gaussians`: 6
- `real-time`: 6
- `localized-editing`: 6
- `reinforcement-learning`: 5
- `dynamic-scene-editing`: 5
- `video-generation`: 5
- `language-embedded-gaussians`: 5
- `semantic-segmentation`: 5
- `open-vocabulary-grounding`: 5
- `scene-flow`: 5
- `real-time-rendering`: 5
- `text-guided-editing`: 5
- `monocular-dynamic-scene`: 5
- `appearance-editing`: 5
- `diffusion-guidance`: 5
- `cross-view-consistency`: 5
- `dataset/kit-ml`: 4
- `repr/smpl-x`: 4
- `Human_Object_Interaction`: 4
- `semantic-grounding`: 4
- `open-vocabulary-query`: 4
- `densification`: 4
- `deformation-mlp`: 4
- `clip-features`: 4
- `dynamic-scene-rendering`: 4
- `motion-aware`: 4
- `geometry-aware-editing`: 4
- `view-consistency`: 4
- `3DGS_Reconstruction`: 4
- `task/physics-based-control`: 3
- `opensource/partial`: 3
- `task/human-object-interaction`: 3
- `Human_Human_Interaction`: 3
- `gaussian-clustering`: 3
- `4d-generation`: 3
- `pose-free`: 3
- `compact-semantic-representation`: 3
- `4d-generator`: 3
- `splatted-features`: 3
- `open-world-segmentation`: 3
- `scene-understanding`: 3
- `point-tracking`: 3
- `motion-tracking`: 3
- `object-removal`: 3
- `self-supervision`: 3
- `compression`: 3
- `explicit-representation`: 3
- `compact-model`: 3
- `wild-reconstruction`: 3
- `static-dynamic-separation`: 3
- `optical-flow-guided`: 3
- `control-points`: 3
- `instructpix2pix`: 3
- `interactive-editing`: 3
- `two-stage-optimization`: 3
- `local-editing`: 3
- `score-distillation`: 3
- `controlnet`: 3
- `image-guided-editing`: 3
- `dataset/motionx`: 2
- `task/motion-editing`: 2
- `dataset/amass`: 2
- `task/motion-captioning`: 2
- `dataset/behave`: 2
- `4d-segmentation`: 2
- `soft-label-segmentation`: 2
- `temporal-identity-field`: 2
- `cross-scene-generalization`: 2
- `multi-view`: 2
- `open-vocabulary`: 2
- `4d-language-field`: 2
- `time-sensitive-query`: 2
- `budget-control`: 2
- `foundation-model-distillation`: 2
- `boundary-guided-refinement`: 2
- `semantic-supervision`: 2
- `3d-masking`: 2
- `motion-scaffold`: 2
- `tri-plane`: 2
- `occlusion-aware`: 2
- `training-free`: 2
- `instance-segmentation`: 2
- `mesh-extraction`: 2
- `multiview-consistency`: 2
- `identity-encoding`: 2
- `multi-view-identity`: 2
- `cross-domain-reference`: 2
- `4d-reconstruction`: 2
- `transformer`: 2
- `optimization-based`: 2
- `relighting`: 2
- `hybrid-representation`: 2
- `geometry-reconstruction`: 2
- `mesh-reconstruction`: 2
- `sparse-control-points`: 2
- `arap`: 2

## Notes

- `norm` is a coarse normalization key for clustering, not necessarily the final canonical tag.
- Final unification should explicitly separate: task tags (directory-aligned), status tags (`status/...`), and technique tags (normalized case/separators).

## Canonical tag policy (current standard)

### 1) Task tags (task/domain categories)

- Canonical task tag = `paperAnalysis/<top-level-directory-name>` (must match the directory name).
- Example: `Motion_Editing`, `Human_Object_Interaction`.
- Hyphenated/lowercase variants (for example `motion-editing`) are normalized back to the directory form.

### 2) Status tags

- Use lowercase hierarchical tags: `status/...` (for example `status/analyzed`, `status/checked`).

### 3) Technique / keyword tags

- Standard rule: lowercase; spaces/underscores -> `-`; keep `/` hierarchy and normalize each segment the same way.
- Example: `VQ-VAE` -> `vq-vae`, `dataset/HumanML3D` -> `dataset/humanml3d`.
