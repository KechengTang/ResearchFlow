---
title: "InteractAvatar: Modeling Hand-Face Interaction in Photorealistic Avatars with Deformable Gaussians"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - avatar
  - hand-face-interaction
  - deformable-gaussians
  - reenactment
  - photorealistic-avatar
  - status/analyzed
core_operator: 在 deformable Gaussians avatar 框架中显式建模 hand-face interaction，通过 Dynamic Gaussian Hand、动态 refinement 与 interaction module 捕获手部细节、接触引起的局部几何/外观变化以及手脸非刚体交互。
primary_logic: |
  先分别建立 photorealistic hand 与 face 的 deformable Gaussian avatar 表示，
  再通过 Dynamic Gaussian Hand 与 dynamic refinement 建模手部 pose-dependent 细节，
  同时引入 hand-face interaction module 捕捉接触带来的几何与外观变化，
  最终实现单目或多视角视频下可新姿态驱动的高保真 hand-face avatar 重建与 reenactment。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_InteractAvatar_Modeling_Hand_Face_Interaction_in_Photorealistic_Avatars_with_Deformable_Gaussians.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:41
updated: 2026-04-18T16:41
---

# InteractAvatar: Modeling Hand-Face Interaction in Photorealistic Avatars with Deformable Gaussians

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2504.07949](https://arxiv.org/abs/2504.07949)
> - **Summary**: InteractAvatar 解决的是 photorealistic avatar 里一个被长期忽视却非常关键的交互细节: 手和脸之间的接触。现有 hand/avatar 方法往往把手和脸分开建模，导致 touching、pressing、遮挡和局部形变都不自然。它因此把 deformable Gaussians 推到了 interaction-aware avatar modeling。
> - **Key Performance**:
>   - 论文在 novel-view synthesis、self-reenactment 和 cross-identity reenactment 上验证其优势。
>   - 同时支持 monocular 和 multi-view 输入下的 hand-face interaction reconstruction。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

真实人类交流里，hand-face interaction 非常常见:

- 扶脸
- 摸下巴
- 手遮脸
- 手带来的局部阴影和皱褶变化

但现有 photorealistic avatar 方法大多忽略这类 interaction。  
InteractAvatar 要解决的是:

**如何在 avatar 表示中真实捕捉手脸接触带来的局部几何、外观和非刚体变化。**

### 核心能力定义

- **输入**: monocular 或 multi-view human interaction videos
- **输出**: 可被 novel pose 驱动的 hand-face interactive avatar
- **强项**: photorealistic interaction、dynamic wrinkles/shadows、reenactment
- **弱项**: 场景更窄，主要服务 avatar 而非通用 4D scenes

### 真正的挑战来源

- 手部细节本身就难建模
- 接触带来的局部压迫、遮挡和阴影变化非常细微
- 若只做刚体 interaction，结果会很假

### 边界条件

- 明显聚焦 hand-face interaction
- 更偏数字人和 avatar 而非普通动态场景
- 依赖 template model 与 deformable Gaussian 组合

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

InteractAvatar 的设计哲学是:

**真实 avatar 不只是独立的人脸和独立的手，还必须显式建模接触交互。**

所以作者不是单独把 face 和 hand 都做好就结束，而是再加 interaction module，专门处理二者之间的非刚体耦合。

### The "Aha!" Moment

真正的 aha 是:

**手脸交互的关键不在于更强的整体身体模型，而在于把 hand-face contact 作为独立的局部动态现象来建模。**

这使模型能显式捕捉:

- fine wrinkles
- complex shadows
- geometry deformation under contact

### 为什么这个设计有效

因为手脸接触的视觉线索高度局部且非刚体。  
如果只靠全局 avatar deformation，通常会被平均掉；  
而 deformable Gaussians + local interaction module 更适合承接这种高频局部变化。

### 对我当前方向的价值

这篇论文对你当前方向的重要性在于:

**Gaussian avatar 不只适合静态 photorealism，也适合 interaction-aware digital human modeling。**

如果后续你考虑表情、接触、身体局部编辑，这篇非常值得复用。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 这其实就是一种 interaction-aware 4D editable avatar 底座，特别适合局部触碰或交互式编辑。
- **与 3DGS_Editing**: 3D avatar editing 往往难以处理接触导致的动态变化，这篇把这种 interaction dynamics 直接纳入表示。
- **与 feed-forward Gaussians**: deformable Gaussian avatar + interaction module 的分解非常适合未来前向 avatar 系统模仿。

### 战略权衡

- 优点: 真实感强，能处理手脸接触这种高难局部现象
- 代价: 任务较专、系统更复杂、数据采集要求更高

---

## Part III / Technical Deep Dive

### Pipeline

```text
monocular or multi-view interaction video
-> deformable Gaussian face/avatar model
-> Dynamic Gaussian Hand with refinement
-> hand-face interaction module
-> reconstruct contact-induced geometry and appearance changes
-> reenactable photorealistic avatar
```

### 关键模块

#### 1. Dynamic Gaussian Hand

手部单独建模，并且支持 pose-dependent 细节变化，这一步本身就比普通手部表示更细。

#### 2. Dynamic Refinement

进一步补充 articulation 过程中出现的细皱褶和复杂阴影变化。

#### 3. Hand-Face Interaction Module

这是整篇的核心。  
它负责处理接触产生的局部几何与外观动态，而不是把 interaction 淹没进整体变形里。

### 关键实验信号

- 论文同时做 self- 和 cross-identity reenactment，说明表示不是只对训练人物过拟合
- monocular 也能做，显示其在更实用设定下仍有价值
- novel-view synthesis 只是基础，interaction realism 才是核心卖点

### 少量关键数字

- 论文核心贡献更偏 qualitative realism 与 reenactment robustness，而非单一指标数值

### 局限、风险、可迁移点

- **局限**: 任务专门，离通用场景 reconstruction 有距离
- **风险**: 接触区域估计若偏差，会直接影响 interaction realism
- **可迁移点**: interaction-aware deformable Gaussians、local contact module、avatar-specific dynamic refinement 都值得迁移

### 实现约束

- 主要面向 photorealistic avatars
- 依赖 hand/face specialized modeling
- 交互细节优先于通用性

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_InteractAvatar_Modeling_Hand_Face_Interaction_in_Photorealistic_Avatars_with_Deformable_Gaussians.pdf]]
