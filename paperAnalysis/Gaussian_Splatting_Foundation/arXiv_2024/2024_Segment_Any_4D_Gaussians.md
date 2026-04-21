---
title: "Segment Any 4D Gaussians"
venue: arXiv
year: 2024
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 4dgs
  - 4d-segmentation
  - temporal-identity-field
  - gaussian-drifting
  - dynamic-scene-editing
  - semantic-grounding
  - status/analyzed
core_operator: 在预训练 4D-GS 之上额外学习一个以 `(Gaussian 位置, 时间)` 为输入的 temporal identity feature field，为每个 Gaussian 预测时变 identity encoding 来抵抗 Gaussian drifting，再结合 refinement 与 Gaussian identity table，把 4D 对象分割变成可缓存、可实时查询、可直接编辑的对象级表示。
primary_logic: |
  先冻结预训练 4D-GS 的 canonical Gaussians 与 deformation field，
  再用 temporal identity feature field 从 `(X, t)` 预测每个 Gaussian 的时变 identity encoding，
  通过 splatting 与 tiny conv decoder 对齐视频跟踪器给出的 2D 伪标签来学习对象身份，
  最后用 4D segmentation refinement 清理离群点和边界歧义，并把每个训练时间戳的对象级高斯集合写入 Gaussian identity table，
  从而实现快速的 4D anything segmentation 与 removal / recoloring / composition。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/arXiv_2024/2024_Segment_Any_4D_Gaussians.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-10T14:26
updated: 2026-04-10T14:26
---

# Segment Any 4D Gaussians

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://jsxzs.github.io/sa4d/) | [arXiv 2407.04504](https://arxiv.org/abs/2407.04504) | [PDF](https://arxiv.org/pdf/2407.04504.pdf) | [Official GitHub](https://github.com/jsxzs/SA4D) | [Reference Code](https://github.com/Marine318/sa4d)
> - **Code Note**: `jsxzs/SA4D` 是论文作者仓库；`Marine318/sa4d` 更像基于论文整理补全的参考实现，可用于复现排查，但不应默认视为官方完整版代码。
> - **Summary**: SA4D 不是把静态 3D segmentation 直接硬搬到 4D-GS，而是显式学习一个随时间变化的 identity field，把“这个 Gaussian 在当前时间属于哪个对象”建模成时变身份问题，从而解决 dynamic 4D scene 里最关键的 Gaussian drifting。之后再把对象级结果缓存成 Gaussian identity table，让分割、mask 渲染和对象编辑都能在秒级完成。
> - **Key Performance**:
>   - 在 HyperNeRF 上，完整方法达到 `89.86% mIoU / 99.24% mAcc`，显著高于 Gaussian Grouping 的 `69.53% / 91.55%`。
>   - 在 Neu3D 上，完整方法达到 `93.02% mIoU / 99.76% mAcc`；引入 Gaussian Identity Table 后，推理渲染速度从 `7.12 FPS` 提升到 `40.36 FPS`，存储仅从 `62.55 MB` 增至 `80.12 MB`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

这篇论文要解决的不是普通的 2D/3D 分割，而是 **dynamic Gaussian scene 里的对象级 4D segmentation**。  
作者希望这个能力同时满足两点：

- 可以像 SAM 一样快速、交互式地选出对象；
- 分割结果在不同时间、不同视角下都保持几何和身份一致。

困难在于，4D-GS 的 canonical Gaussians 经过 deformation 后，某些 Gaussian 在不同时间点可能会“漂”到另一个对象上。作者把这个问题称为 **Gaussian drifting**。如果仍然采用静态 3D 方法里“每个 Gaussian 绑定一个固定 identity feature”的做法，那么被第一帧选中的高斯，到了后续时间可能已经不再代表同一个真实对象。

### 核心能力定义

- **输入**：预训练好的 4D Gaussian scene、训练视频、视频跟踪器输出的逐帧对象 identity mask、用户给定的 object ID。
- **输出**：每个时间戳上属于目标对象的 4D Gaussians，以及 novel view / novel timestamp 下的 anything masks。
- **直接能力**：对象级 4D segmentation、anything mask rendering、对象 removal、recoloring、composition。

### 真正的挑战来源

- **语义不是静态的**：同一 canonical Gaussian 在不同时间的对象归属可能改变。
- **监督又稀疏又脏**：作者没有 4D GT object labels，只能依赖视频跟踪 foundation model 的 2D pseudo masks。
- **动态场景是非刚性的**：直接把 3D segmentation 结果通过 deformation field 传播到别的时间，经常会出现身份错配。
- **交互速度也很重要**：如果每次查询都要重新跑隐式 field 和逐帧后处理，就很难真正用于编辑。

### 边界条件

- 该方法建立在预训练 4D-GS 上，本身不是重建方法，而是一个挂在 4D 表示上的 segmentation / editing layer。
- 它当前使用的是 **object ID prompt**，不是 click-based 或 language-based prompt。
- 多视角视频里如果 video tracker 给同一对象分配了不同 ID，会带来 identity collision；因此论文在 Neu3D 上只用单个相机视角训练语义分支。
- refinement 依赖 2D tracker mask 作为先验，因此 tracker 出错时，边界修正也可能失败。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

SA4D 的核心设计不是“把 2D mask 提升到 4D”，而是把 4D segmentation 重新定义为：

**在 deformation-based 4D Gaussian representation 中，持续识别同一个真实对象的时变 Gaussian 子集。**

这意味着作者没有把语义看成 canonical space 里的固定标签，而是把它建模成 **time-conditioned identity encoding**。也就是说：

- 几何还是由预训练 4D-GS 的 canonical Gaussians + deformation field 负责；
- 语义则由一个额外的 temporal identity feature field 负责；
- 编辑和渲染时再通过 identity table 读取对象级 Gaussians。

这种解耦让 SA4D 不用重新训练 4D 几何表示，只需要在固定的 4D-GS 上学一个轻量的时变身份层。

### The "Aha!" Moment

真正的 aha moment 是：

**4D 里的对象分割难点不在“怎么给每个 Gaussian 一个标签”，而在“怎么让这个标签随时间仍然对应同一个真实对象”。**

因此作者放弃了静态 identity embedding，改为学习：

$$
e = \phi_\theta(\gamma(X), \gamma(t))
$$

其中输入是 Gaussian 在 canonical space 的位置和时间，输出是该时刻的 identity encoding。  
这一步直接把 Gaussian drifting 暴露成一个可学习的时变身份建模问题，而不是留给 deformation field 自己“顺便解决”。

### 为什么这样有效？

原因有三层：

1. **语义和几何解耦**  
   4D-GS 负责时空几何与渲染，TFF 只负责对象身份，避免在重建阶段把语义目标硬塞进 radiance optimization。

2. **从 noisy 2D supervision 中做时间聚合**  
   视频跟踪器给的是逐帧 noisy masks。TFF 把这些 sparse、脏的身份信号投回 canonical Gaussians，并通过时间条件化建模实现跨时间聚合。

3. **把“实时可用性”显式工程化**  
   训练完成后，作者并不在推理时持续跑隐式网络，而是把每个训练时间戳上的对象级结果写入 Gaussian identity table，再做最近邻时间插值。这相当于把“语义查询”从 online implicit decoding 改成了 cached object retrieval。

### 策略权衡

- **优点**：速度快、对 Gaussian drifting 更稳、对象级编辑接口直接。
- **代价**：语义层仍依赖 tracker 伪标签质量；而且 deformation field 不是 object-level 可分解的，所以渲染时还得带着整个 deformation 网络一起工作。
- **适用场景**：特别适合“先有 4D scene，再往上加语义选择和对象编辑”的 research line。

## Part III / Technical Deep Dive

### Pipeline

```text
pretrained 4D-GS
-> canonical Gaussians + deformation field
-> temporal identity feature field predicts time-varying identity encodings
-> splat identity features to 2D identity map
-> tiny conv decoder + CE/KL losses align with video-tracker masks
-> per-object Gaussian sets at training timestamps
-> refinement removes outliers and ambiguous boundary Gaussians
-> Gaussian identity table caches results
-> real-time 4D segmentation / removal / recoloring / composition
```

### 关键模块

#### 1. Temporal Identity Feature Field

TFF 是一个很轻量的 MLP。补充材料给出的结构是：

- 输入：`γ(X)` 与 `γ(t)`；
- 主干：3 层全连接 + ReLU，每层 256 channels；
- 输出：先得到 256 维 feature，再映射到 32 维 identity encoding。

这个设计点很重要，因为它表明作者没有试图学习复杂的高维 semantic field，而是只学习足够支持 object identity 的紧凑表示。

#### 2. 2D 伪标签监督与 3D 正则

作者用 video tracker 产生逐帧 identity masks 作为 2D supervision。  
训练目标由两部分组成：

- `L2d`：标准 cross-entropy，约束渲染到图像平面的 identity prediction 与 tracker mask 对齐；
- `L3d`：对 deformed space 里近邻 Gaussian 的 identity encoding 加 KL 正则，减少被遮挡区域和稀疏监督下的噪声。

这个组合的作用是：  
先让模型学会“像素上是谁”，再让它在 3D Gaussian 邻域里保持局部身份一致性。

#### 3. 4D Segmentation Refinement

单靠 TFF 得到的结果仍然会有两类问题：

- 远处离群 Gaussians；
- 物体边界附近语义模糊、颜色相似的 ambiguous Gaussians。

因此作者设计了两步 refinement：

1. 先做 outlier removal；
2. 再用 2D mask prior 做 projection loss，把边界上负梯度的 Gaussians 删掉。

这一步的本质是：  
先用 4D identity field 得到一个语义粗解，再用 2D mask 先验把对象几何修薄、修干净。

#### 4. Gaussian Identity Table

这是论文里非常值得记住的工程 operator。  
作者发现，如果每次推理都现算隐式 identity field，再逐帧 refinement，会拖慢交互。因此他们把每个训练时间戳上的分割结果预先存进一个 **Gaussian identity table**，推理时只做最近邻时间插值。

这个缓存层带来的收益非常直接：

- 推理更快；
- 编辑更方便；
- 对象级操作接口从“隐式 field 查询”变成“表查 + 高斯集合操作”。

### 关键实验信号

- **整体性能明显超过 3D baselines**：  
  在 HyperNeRF 上，SA4D 完整版本 `89.86% mIoU / 99.24% mAcc`，对比 SAGA 的 `65.25 / 75.56` 和 Gaussian Grouping 的 `69.53 / 91.55`，提升非常明显。

- **TFF 对 Gaussian drifting 是真的有效**：  
  在最典型的 `split-cookie` 场景里，不加 TFF 时 IoU 为 `76.89%`，加了 TFF 但不做 refinement 就到 `83.31%`，完整版本进一步到 `88.92%`。这说明提升不只是后处理带来的，而是时变 identity 建模本身确实在修正跨时间身份漂移。

- **Identity table 让系统从“能跑”变成“能交互”**：  
  不使用 GIT 时只有 `7.12 FPS`，引入后达到 `40.36 FPS`。这正是作者所说“within 10 seconds 完成对象分割，然后以接近原 4D-GS 的速度继续渲染”的关键支撑。

- **Refinement 的收益真实但有 trade-off**：  
  时间间隔从 `1` 增到 `16` 时，refinement 耗时从 `7.8s` 降到 `0.48s`，但 IoU 从 `88.92%` 降到 `87.58%`。说明这套后处理可以根据交互速度和精度需求做调节。

### 少量关键数字

- 训练设备：单张 `RTX 3090`
- 训练轮数：`5000` iterations
- 学习率：`5e-4`
- 典型场景训练成本：约半小时，约 `200k` Gaussians，分辨率 `1352 x 1014`
- Identity encoding 维度：`32`

### 对你可迁移的 operator

- **time-varying identity field instead of static semantic feature**
- **freeze geometry / learn semantics later**
- **semantic object cache on top of 4DGS via identity table**
- **use 2D priors only for boundary cleanup instead of full semantic reconstruction**

如果你后面想做 4DGS 里的语义 grounding、交互式对象编辑、或面向编辑的 object decomposition，这篇论文最值得迁移的不是某个 loss，而是它的分层方式：

**先把 4D reconstruction 稳定好，再把对象身份作为一个独立的时变场挂上去，最后用一个可缓存的对象表把在线交互做实。**

### 实现约束与局限

- prompt 仍然是 object ID，不够自然；
- deformation field 不是按对象拆开的，意味着对象级操作还无法完全摆脱全局 deformation；
- 多视角视频中存在 mask identity collision，限制了多视角联合语义训练；
- refinement 会受 video tracker 的误分割影响，论文补充材料里展示了把整只杯子误删的失败案例。

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/arXiv_2024/2024_Segment_Any_4D_Gaussians.pdf]]
