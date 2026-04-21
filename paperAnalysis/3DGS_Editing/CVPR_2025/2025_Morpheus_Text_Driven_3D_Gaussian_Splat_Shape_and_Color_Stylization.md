---
title: "Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization"
venue: CVPR
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - text-guided-editing
  - stylization
  - shape-editing
  - color-editing
  - rgbd-diffusion
  - controlnet
  - autoregressive
  - depth-guided-feature-sharing
  - status/analyzed
core_operator: 把 3DGS stylization 改写成 autoregressive RGBD diffusion pipeline，用可分别控制 shape 与 appearance 的 RGBD 去噪、Warp ControlNet 和 depth-guided feature sharing 在逐帧 stylization 时维持多视角一致性，最后再回训出新的 stylized 3DGS。
primary_logic: |
  输入原始 3DGS、文本 prompt 与 shape/appearance strength 后，先沿代表轨迹渲染 RGBD 序列，
  再用带双强度控制的 RGBD diffusion stylize 第一帧，并把先前 stylized 帧 warp 到当前视角形成 composite 作为 Warp ControlNet 条件，
  同时用 depth-informed cross-attention 与 feature injection 传播高层 stylization 语义，最后以生成的 stylized RGBD 帧重新训练 3DGS 得到最终可渲染的 stylized scene。
pdf_ref: paperPDFs/3DGS_Editing/CVPR_2025/2025_Morpheus_Text_Driven_3D_Gaussian_Splat_Shape_and_Color_Stylization.pdf
category: 3DGS_Editing
created: 2026-04-21T16:00
updated: 2026-04-21T16:00
---

# Morpheus: Text-Driven 3D Gaussian Splat Shape and Color Stylization

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://nianticlabs.github.io/morpheus/) | [CVPR 2025 Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wynn_Morpheus_Text-Driven_3D_Gaussian_Splat_Shape_and_Color_Stylization_CVPR_2025_paper.pdf)
> - **Summary**: Morpheus 的关键不是只把 3DGS 当作一堆待贴风格纹理的视图，而是把 stylization 做成一个 RGBD-aware、可分别控制几何与外观强度的 autoregressive 过程，因此它能比多数 3DGS stylization 方法更明显地改 shape，同时维持多视角一致性。
> - **Key Performance**:
>   - 在 `53` 组 stylization 上，`CLIP Direction Similarity / Consistency / RMSE / LPIPS` 分别达到 `0.175 / 0.606 / 0.0370 / 0.0378`，优于 `GaussCtrl` 的 `0.123 / 0.590 / 0.0471 / 0.0438`。
>   - 用户研究使用 `31` 位参与者，论文报告在 style adherence 上明显优于 baselines，在 aesthetic quality 上至少匹配并常常更好。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

Morpheus 针对的是 3DGS stylization 里一个很具体但很难的问题：

**现在很多 novel-view stylization 方法能改颜色和纹理，但很难在保持多视角一致的同时，真正改动 scene 的几何感和形状感。**

这件事难的根源在于：

- 2D diffusion 本身没有显式几何意识。
- 一旦你想在多个视角上改 shape，多视角 consistency 会马上变脆。
- 只做单帧 stylization 再回训 3D 表示，往往只能得到表层 texture 变化。

### 核心能力定义

- **输入**：一个深度正则化的 3DGS、文本 prompt、appearance strength、shape strength。
- **输出**：一个新的 stylized 3DGS。
- **能做**：text-driven scene stylization，且同时控制 shape / color 的改动强度。
- **不主打**：实时交互式编辑；它仍然是一个离线约 `10` 分钟量级的 pipeline。

### 真正的挑战来源

- 跨视角传播 stylization 时，深层语义和局部纹理很容易错位。
- 如果只 warp RGB，不处理深度和 feature-level correspondence，错误会积累成重复纹理或几何错位。
- 若 shape 和 appearance 用同一 noise schedule 去改，往往会纠缠在一起，控制不清楚。

### 边界条件

Morpheus 假设底层 3DGS 已经足够好，并依赖代表性轨迹的 RGBD renders。它解决的是 stylized 3D scene creation，而不是任意局部 object editing。

## Part II / High-Dimensional Insight

### 方法真正新的地方

Morpheus 把 3DGS stylization 的任务改写成三件事：

1. **RGBD jointly stylize**
2. **shape / appearance strength disentangle**
3. **autoregressive multi-view propagation**

相比只在 RGB 上做风格迁移，它明确把 depth 也拉进来了。

### The "Aha!" Moment

真正的 aha 是：

**如果你想在 3DGS 上做明显的 shape stylization，就不能只传播像素颜色，而必须把“几何约束下的 stylization 信息”也一起沿视角序列传递。**

于是 Morpheus 的设计链路是：

- 先对 RGBD 一起做 stylization
- 再通过 warped composite + Warp ControlNet 把之前 stylized 结果传播到新视角
- 再通过 depth-informed feature sharing 精细传语义，而不是让 reference frame 全图无脑 cross-attend

这使它的 stylization 不再是“每帧都各画各的”，而更像是“在 geometry-aware 轨迹上持续生长的风格化场景”。

### 为什么这个设计有效？

- 深度通道让 diffusion 知道哪些地方真的应该发生 shape change。
- 分开控制 `T^D_max` 和 `T^I_max`，可以把 geometry stylization 和 appearance stylization 的强度拆开。
- Warp ControlNet 不只是做补洞，而是把 warped history 变成当前帧 stylization 的结构化先验。

### 战略权衡

- **优点**：可控地改颜色和形状，且比很多现有方法更不容易只停留在表层 texture。
- **代价**：仍是逐轨迹的 autoregressive stylization；而且质量依赖原始 3DGS 和深度预测质量。

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS + prompt + shape/appearance strength
-> render representative RGBD trajectory
-> RGBD diffusion stylizes first frame with separate shape / color schedules
-> warp previous stylized RGBD frames to current view
-> build composite RGBD + validity mask
-> Warp ControlNet guides current-frame RGBD stylization
-> depth-guided cross-attention + feature injection propagate stylization semantics
-> retrain a new stylized 3DGS on stylized RGBD frames
```

### 关键机制

#### 1. RGBD stylization with separate strength control

Morpheus 不是在 RGB 上做一个普通 text stylization，而是让 RGB 和 depth 一起去噪，并用两套最大噪声步数分别控制 shape 和 appearance 的改动强度。这一点对你后续想拆 `appearance vs geometry edit variable` 很有启发。

#### 2. Warp ControlNet

作者没有直接把上一帧 stylized image 硬 warp 过来然后 inpaint，而是先构造 composite，再训练一个专门处理 warped RGBD 条件的 ControlNet 来纠正错位和补足缺失区域。

#### 3. Depth-guided feature sharing

它没有让 reference frames 对当前帧做全图平均 cross-attention，而是用 depth 约束哪些区域该互相看见、哪些 feature 该注入。这避免了很多多视角 stylization 常见的“纹理复制”与错位问题。

### 关键实验信号

- 在总体 quantitative benchmark 上，Morpheus 的 CLIP-based style following、consistency 和 image consistency 指标都优于 Instruct NeRF2NeRF、Instruct GS2GS、GaussCtrl 和 DGE。
- 论文最重要的 qualitative 信号是：它能在保持 scene coherence 的同时明显改动 shape，而不是只做局部颜色涂抹。
- 用户研究也支持这个结论：它在 prompt adherence 上更强，在 aesthetic quality 上至少不输 baselines。

### 对你最有价值的启发

对你现在的 `A0 localized appearance/material edit`，Morpheus 不是直接 teacher，但它提供了两个很值钱的 operator：

- appearance / geometry strength disentanglement
- depth-aware feature propagation

如果你未来要把 A0 往 mild geometry 或更强 temporal propagation 扩展，这两个点都很值得借。

## Local Reading / PDF 参考

![[paperPDFs/3DGS_Editing/CVPR_2025/2025_Morpheus_Text_Driven_3D_Gaussian_Splat_Shape_and_Color_Stylization.pdf]]
