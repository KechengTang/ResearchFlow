---
title: "3D Gaussian Inpainting with Depth-Guided Cross-View Consistency"
venue: CVPR
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - inpainting
  - object-removal
  - cross-view-consistency
  - depth-guided-mask
  - reference-view-guidance
  - scene-editing
  - status/analyzed
core_operator: 先用多视角深度把每个视角的待补区域缩成“所有视角都不可见”的真正空洞，再用单参考视角 2D inpainting 结果回投成跨视角监督，联合优化 object-removed 3DGS。
primary_logic: |
  输入源 3DGS、多视角图像和待移除对象的 masks，
  先通过各视角深度与相机位姿把其他视角可见背景投回当前视角，得到仅包含真实不可见区域的 refined inpainting mask，
  再在参考视角上做 2D RGB 与 depth inpainting，
  最后把该参考视角的补洞结果投回其余视角作为 cross-view supervision，联合优化新的 3DGS 以实现一致补洞。
pdf_ref: paperPDFs/3DGS_Editing/CVPR_2025/2025_3D_Gaussian_Inpainting_with_Depth_Guided_Cross_View_Consistency.pdf
category: 3DGS_Editing
created: 2026-04-18T17:05
updated: 2026-04-18T17:05
---

# 3D Gaussian Inpainting with Depth-Guided Cross-View Consistency

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_3D_Gaussian_Inpainting_with_Depth-Guided_Cross-View_Consistency_CVPR_2025_paper.pdf)
> - **Summary**: 3DGIC 真正抓住的问题不是“怎么补洞”，而是“哪些区域其实不用补”。它通过深度和多视角可见性，把每张图的 inpainting mask 缩减到跨视角都看不到的真实空洞，再用一个参考视角的 inpainting 结果去监督所有视角，因此比按视角分别补洞的方法更稳定、更一致。
> - **Key Performance**:
>   - 使用 LDM 作为 2D inpainter 时达到 `FID 36.4 / m-FID 96.3 / LPIPS 0.26 / m-LPIPS 0.028`，优于 `GScream 38.6 / 101.6 / 0.28 / 0.033`。
>   - 即便使用非扩散的 `LaMa`，仍优于 `MVIP-NeRF` 与多数旧基线，说明提升不只来自更强的 2D inpainter。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文的核心问题非常具体：

**当前 3D inpainting 里很多视角其实把“本来在别的视角可见的背景”也错当成待补区域，于是不同视角会各自脑补出不同内容，最后 3D 一致性被破坏。**

所以 3DGIC 的重点不是先换个更强的 2D inpainting 模型，而是先重新定义：

**每个视角真正应该被 inpaint 的区域到底是什么。**

### 核心能力定义

- **输入**：源 3DGS、多视角图像、待删除对象的 masks。
- **输出**：移除目标对象并完成跨视角一致补洞的 3DGS。
- **擅长**：object removal + scene inpainting。
- **主要收益**：减少因 per-view mask 过大造成的跨视角不一致。

### 真正的挑战来源

- 许多 per-view mask 覆盖了“在当前视角被挡住、但在其他视角可见”的背景区域。
- 如果每个视角都独立补这些区域，就会引入不一致纹理和几何。
- 直接依赖所有视角各自的 2D inpainting 结果作为监督，会把这些矛盾传播进 3DGS。

### 边界条件

- 需要已有的待移除对象 masks，可由 SAM 等方法提供。
- 依赖一个质量尚可的初始 3DGS 和其渲染深度。
- 仍是以单参考视角为主的引导方案，不是所有视角同时直接生成。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

3DGIC 的哲学很清晰：

**先用几何可见性裁掉“假空洞”，再用参考视角补“真空洞”。**

与大量方法把注意力放在“如何让不同视角补得更像”不同，3DGIC 先往前退一步，问的是：

**为什么这些视角需要补的区域会不同？**

答案就是：很多区域其实根本不该补，只是 mask 太粗。

### The "Aha!" Moment

真正的 aha 是：

**用其他视角已经看见的背景，去反向剔除当前视角的 inpainting 区域。**

作者通过各视角的 depth 和 camera pose，把其他视角里可见的背景投影回当前视角。凡是能被其他视角提供真实背景的部分，就从当前视角的 inpainting mask 中删掉。最后留下的才是跨视角都不可见、确实需要生成的区域。

### 为什么这个设计有效？

- 它把 per-view inpainting 从“生成缺失内容”转化为“只生成真正没有被任何视角观测到的内容”。
- mask 一旦变得更干净，2D inpainting 的负担和不确定性都会明显下降。
- 作者进一步只在参考视角做 2D RGB + depth inpainting，再把结果投回其他视角，等价于给整个 3D 优化提供一个统一的补洞源，而不是让每个视角自己发散。

### 战略权衡

- **优点**：用几何可见性直接减少了多视角 2D inpainting 的冲突源。
- **代价**：比较依赖深度质量、相机位姿和参考视角的选择。
- **定位**：这是一个面向 object removal / inpainting 的几何一致性补洞框架，不是通用文本编辑器。

## Part III / Technical Deep Dive

### Pipeline

```text
source 3DGS + multi-view images + object masks
-> render depth maps from source 3DGS
-> project visible backgrounds across views
-> refine each view's inpainting mask to keep only truly invisible regions
-> remove object Gaussians and add random Gaussians in masked region
-> choose the view with the largest refined mask as reference
-> 2D inpaint RGB and depth on reference view
-> project reference inpainted content to other views
-> optimize new 3DGS with rendering loss + cross-view LPIPS loss
```

### 关键模块

#### 1. Inferring Depth-Guided Inpainting Masks

这是整篇论文最关键的模块。作者并不直接接受原始 object mask，而是把各视角可见背景互相投影，更新每个视角的 mask。最终 refined mask 只保留“在所有训练视角都不可见”的区域，从源头减少了错误补洞。

#### 2. Reference-view inpainting

参考视角不只是用来生成一张好看的 2D 图，而是作为整个 3D 补洞的统一锚点。作者对参考视角的 RGB 和 depth 都做 2D inpainting，这样后续回投到其他视角时，不只有纹理信息，还有几何约束。

#### 3. Cross-view consistent loss

把参考视角 inpaint 后的区域投回其他视角，构造 `I^P_k` 作为监督，再用 `LPIPS` 约束各视角渲染与这个统一 supervision 一致。这样 3DGS 学到的是单一、共享的补洞解释，而不是多视角各自为政。

### 关键实验信号

- 在 SPIn-NeRF 数据集上，3DGIC 的两个版本都优于多数基线，其中 LDM 版本在四项指标上全部最好。
- 论文明确指出，提升并不只来自更强的 2D inpainter，因为 LaMa 版本同样有明显优势。
- 定性上，相比 `Gaussian Grouping`、`GScream` 等方法，3DGIC 更能保留原场景中的可见背景细节，同时避免跨视角 logo / 纹理不一致问题。

### 少量关键数字

- `3DGIC + LDM`: `FID 36.4`, `m-FID 96.3`, `LPIPS 0.26`, `m-LPIPS 0.028`
- `3DGIC + LaMa`: `FID 41.7`, `m-FID 102.4`, `LPIPS 0.28`, `m-LPIPS 0.032`
- `GScream`: `FID 38.6`, `m-FID 101.6`, `LPIPS 0.28`, `m-LPIPS 0.033`

### 实现约束

- 训练时选择 refined mask 最大的视角作为 reference view，因为它覆盖最多待补 3D 空间。
- 总损失由 reference-view rendering loss 与 multi-view cross-view LPIPS 共同组成。
- 结果可在优化完成后用于任意相机位姿的新视角渲染。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/CVPR_2025/2025_3D_Gaussian_Inpainting_with_Depth_Guided_Cross_View_Consistency.pdf]]
