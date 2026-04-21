---
title: "3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing"
venue: ICLR
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - drag-based-editing
  - diffusion-guidance
  - control-points
  - geometry-aware-editing
  - cross-view-consistency
  - status/analyzed
core_operator: 用 3D handle/target 点先对局部 Gaussians 做 copy-and-paste 式粗形变，再用 LoRA 微调的 diffusion 对多视角渲染做 annealed image-to-image 校正，并通过多步调度与 handle relocation 支持更大幅度的 drag 编辑。
primary_logic: |
  输入已训练 3DGS 与若干 3D handle/target 点对，
  先把拖拽拆成多个中间目标步，并对 handle 邻域的 Gaussians 做插值位移与旋转形变，同时保留原高斯的低透明度副本，
  再渲染多视角图像并用 LoRA 微调的 diffusion 执行 view correction，把伪影与错误语义修回一致的 2D 监督，
  最后在局部 mask 内重新优化 3DGS，并在每一步结束后重定位 handle、更新历史图像缓冲区，得到多视角一致的几何拖拽结果。
pdf_ref: paperPDFs/3DGS_Editing/ICLR_2025/2025_3DGS_Drag_Dragging_Gaussians_for_Intuitive_Point_Based_3D_Editing.pdf
category: 3DGS_Editing
created: 2026-04-18T18:10
updated: 2026-04-18T18:10
---

# 3DGS-Drag: Dragging Gaussians for Intuitive Point-Based 3D Editing

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/cde7c4434681c3988f9c5870c87cdd10-Paper-Conference.pdf) · [GitHub](https://github.com/Dongjiahua/3DGS-Drag)
> - **Summary**: 3DGS-Drag 试图把 2D drag editing 那种“点一下再拖到目标位置”的直觉交互，真正变成可用于真实 3D 场景的几何编辑接口。它不再只靠文本提示改外观，而是用 3D 控制点先给出粗几何，再让 diffusion 修复内容与纹理，因此能同时做动作变化、形状拉伸、局部补洞和内容延展。
> - **Key Performance**:
>   - 在量化评价中，GPT 评分达到 `Editing Effectiveness 3.67 / Rendering Quality 3.47`，明显高于 `Instruct-NeRF2NeRF` 与 `PDS`。
>   - 用户偏好为 `77.90%`，单次编辑约 `15 min`，在单张 `RTX 4090` 上明显快于 `Instruct-NeRF2NeRF` 的约 `1 h`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文聚焦的是一个此前在 3DGS 里并没有被很好解决的能力：

**如何像 2D DragGAN 一样，用少量点级约束去做直观、精细、带几何变化的 3D 场景编辑。**

此前两类路线都不够理想：

- 纯 deformation-based 方法往往依赖强几何先验、控制图或动态视频，不适合随手拖拽真实场景。
- 纯 2D diffusion 编辑虽然能补内容，但遇到较大几何变化时很难稳定回写到 3D，容易跨视角不一致。

3DGS-Drag 的切入点很明确：把“拖拽”本身作为主接口，把 3D 控制点直接放进场景编辑回路。

### 核心能力定义

- **输入**：已重建的 3D Gaussian scene，以及一组 3D handle points 和对应的 3D target points。
- **输出**：满足拖拽意图、且多视角一致的编辑后 3DGS。
- **擅长**：姿态调整、局部形状拉伸、物体平移、背景补洞、内容延展。
- **不擅长**：完全不可见区域的大幅结构生成，以及超出大多数视角可见范围的极端拖拽。

### 真正的挑战来源

- 只有少量控制点时，真实形变函数不可直接求得，只能先给出近似几何引导。
- 直接移动高斯会产生错误纹理、洞和语义伪影，仅靠形变本身无法生成合理内容。
- 对长距离 drag，如果一步做到位，diffusion 很容易失去结构约束，训练也更难收敛。

### 边界条件

- 方法默认已有质量不错的静态 3DGS。
- 编辑严重依赖目标区域在多视角中仍有足够观测。
- 对“拖到视野外”或“生成全新背面”这类任务，论文明确仍会失败。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

3DGS-Drag 的设计不是“让 diffusion 直接学会 3D 拖拽”，而是把任务拆成两个互补模块：

- **deformation guidance** 先解决几何往哪走；
- **diffusion guidance** 再解决拖过去以后该长成什么样。

这一步很关键，因为作者并不把 3DGS 当成一个最终可精确形变的几何体，而是把它当成一个可被粗拖动、再被 2D 先验修复的中间表征。

### The "Aha!" Moment

真正的 aha 是：

**粗形变并不需要一开始就完全正确，它只需要把多视角编辑拉到一个更容易被 diffusion 一致修复的空间。**

所以作者选择了一个非常务实的组合：

- 对 handle 邻域高斯做插值平移和近似旋转，先把局部结构“拖过去”；
- 不直接覆盖原高斯，而是 copy-and-paste 新高斯并降低旧高斯透明度，给后续优化留下回旋余地；
- 再用 LoRA 微调后的 diffusion 做 inverse-free 的 image-to-image 校正，把错误语义和伪影修回去。

换句话说，3DGS-Drag 真正成功的原因不是某个单独模块特别强，而是它让几何引导和语义修复分别负责自己最擅长的那一部分。

### 为什么这个设计有效？

- 3DGS 的显式高斯表示使局部拖动和 mask 渲染都很方便，适合作为交互式控制底座。
- deformation 把不同视角的编辑目标先对齐到相近结构，降低了 2D diffusion 的随机性和语义漂移。
- diffusion 再负责补回拉伸后的纹理、遮挡区和新暴露区域，从而避免纯形变方法的“空壳几何”问题。
- progressive scheduler 把长距离编辑拆成多个小步，显著降低了一步编辑时的结构崩坏风险。

### 战略权衡

- **优点**：交互接口自然，几何编辑能力明显强于纯文本驱动方法。
- **代价**：仍要依赖 2D diffusion 教师，且完整编辑时间仍是分钟级，不是实时预览。

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS scene + 3D handle/target points
-> multi-step editing scheduler generates intermediate targets
-> local Gaussian deformation with translation/rotation interpolation
-> copy-paste deformed Gaussians + reduce source opacity
-> render selected views with local masks
-> LoRA-tuned diffusion performs image-to-image correction
-> optimize 3DGS with clean local supervision
-> relocate handles and update image buffer
-> final drag-edited 3DGS
```

### 关键模块

#### 1. Deformation Guidance

作者把每个 handle 点周围一定距离内的高斯分配给该点，再用最近的若干 handle 点去插值高斯的平移和旋转。  
这一步给出的是一个**够用但不追求完美**的粗形变，其目标是提供几何引导而不是直接输出最终结果。

#### 2. Local Editing Mask

论文同时维护 3D mask 和每视角 2D mask，只在局部区域更新目标内容，并尽量保住背景。  
这点很重要，因为没有 local mask 时，作者的消融显示全场景会变糊，拖拽也容易失败。

#### 3. Diffusion View Correction

作者没有采用 DragDiffusion 这类需要反演的慢流程，而是对 deformed rendering 做 image-to-image 校正。  
LoRA 微调让 diffusion 更贴合当前场景，因此它更像一个多视角一致的局部修复器，而不是自由发挥的重绘器。

#### 4. Annealed Dataset Editing

论文不是像 IN2N 那样长期迭代式编辑数据集，而是只进行有限次数更新，并逐步减小 image-to-image strength。  
这相当于先做粗修改，再做细修，减少几何不一致带来的训练震荡。

#### 5. Multi-step Editing + Handle Relocation

每一步结束后，作者用被拖拽高斯的平均位移去更新 handle 位置，并把新编辑结果加入 diffusion fine-tuning buffer。  
这保证了下一步拖拽总是从最新局部几何出发，而不是反复基于最初状态硬拉。

### 关键实验信号

- 与 `Instruct-NeRF2NeRF`、`PDS`、纯 deformation、逐视角 `SDE-Drag` 相比，3DGS-Drag 在图 6 中是唯一能同时做好几何变化和内容修补的方法。
- 没有 local editing 时，背景显著变糊；只做 `1-step` drag 时，大幅拖拽会明显破碎，说明 progressive schedule 不是可有可无。
- 量化上，`3.67 / 3.47` 的 GPT 评分和 `77.90%` 的用户偏好，说明它不只是“能拖动”，而是在编辑质量上也更稳定。

### 方法边界与风险

- 看不见的背面和边界外拖拽是明确失败案例。
- 目标太小、场景过大或过复杂时，diffusion guidance 仍会不稳。
- 本质上它仍是“粗几何 + 2D 修复”的体系，不是显式 3D 生成模型。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICLR_2025/2025_3DGS_Drag_Dragging_Gaussians_for_Intuitive_Point_Based_3D_Editing.pdf]]
