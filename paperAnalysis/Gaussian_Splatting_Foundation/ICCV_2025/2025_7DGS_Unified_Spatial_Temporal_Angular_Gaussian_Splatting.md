---
title: "7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 7dgs
  - view-dependent-effects
  - dynamic-scene-rendering
  - conditional-slicing
  - unified-representation
  - status/analyzed
core_operator: 用 7D Gaussians 同时覆盖空间位置、时间和视角方向，并通过 conditional slicing 将 7D primitive 在给定时间与视角下切成兼容现有 3DGS 渲染管线的 3D Gaussians，从而统一动态建模与 view-dependent appearance。
primary_logic: |
  先将场景元素表示为 spanning position、time 和 viewing direction 的 7D Gaussians，
  再在给定时间戳和观察方向时通过 conditional slicing 得到对应的 3D Gaussians，
  随后沿用现有 3DGS pipeline 进行渲染和优化，
  最终在实时条件下统一处理动态变化和 view-dependent effects。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_7DGS_Unified_Spatial_Temporal_Angular_Gaussian_Splatting.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:33
updated: 2026-04-18T16:33
---

# 7DGS: Unified Spatial-Temporal-Angular Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2503.07946](https://arxiv.org/abs/2503.07946) · [Project Page](https://gaozhongpai.github.io/7dgs/)
> - **Summary**: 7DGS 想补的是 Gaussian 表示里一个长期分裂的问题: 动态场景通常交给 4DGS，视角相关外观通常交给 6DGS 或更复杂外观模块，但两者很少真正统一。它因此把空间、时间和视角一起放进 7D Gaussian 里，再通过 conditional slicing 回落到 3DGS 渲染接口。
> - **Key Performance**:
>   - 论文报告在复杂动态且具 view-dependent effects 的场景上可比 prior methods 高最多 `7.36 dB` PSNR。
>   - 同时保持约 `401 FPS` 的实时渲染。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

动态场景渲染里常有两个分开的难点:

- 场景随时间变化
- 外观随视角变化

很多方法只能抓住其中一半。  
7DGS 要解决的是:

**能不能在同一套 Gaussian 表示里，把 temporal dynamics 和 view-dependent appearance 一起统一建模。**

### 核心能力定义

- **输入**: 具有动态变化和视角相关外观的场景
- **输出**: 同时支持时间和视角条件的实时渲染表示
- **强项**: unified representation、view-dependent dynamic rendering、实时性
- **弱项**: 表示维度更高，训练和解释都更复杂

### 真正的挑战来源

- 时间和视角两个维度彼此耦合
- moving specular highlight、translucency 等现象很难只靠 4D 或只靠 angular field 解决
- 一旦维度提升过高，实时渲染容易崩

### 边界条件

- 方法更适合高保真 dynamic appearance，而不是纯几何型任务
- 依赖条件切片和现有 3DGS pipeline 的兼容性
- 不是最简洁的 4DGS baseline，而是更高维的统一表示

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

7DGS 的设计哲学是:

**与其在 4D 表示上外挂视角模块，不如从表示层直接把时间和视角并入 Gaussian primitive。**

这意味着 7D primitive 本身就承载了:

- 3D position
- 1D time
- 3D viewing direction

### The "Aha!" Moment

真正的 aha 是:

**高维表示不一定意味着高维渲染，只要有一个高效的 conditional slicing，就能把 7D 表示在运行时切回普通 3DGS 可处理的形式。**

这让作者避免了完全重写渲染器，同时获得 unified modeling 的好处。

### 为什么这个设计有效

时间和视角相关外观本来就是同一个现象的两个条件变量。  
把它们统一进 primitive 后，模型可以直接学习这两个维度的联合分布，而不是靠多个模块事后拼接。

### 对我当前方向的价值

对你当前方向，这篇很重要，因为它回答了一个 foundation-level 问题:

**Gaussian primitive 的维度还能继续扩，且扩维不一定破坏实时性。**

如果你后面做语言、高光、材料、时间统一建模，这类 higher-dimensional Gaussian 很值得参考。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 一旦编辑目标涉及 view-dependent material 或 specular behavior，单纯 4D 编辑不够，7D unified primitive 更像可编辑底座。
- **与 3DGS_Editing**: 3D 编辑通常只处理静态视角相关外观，7DGS 则把时间也纳入编辑空间。
- **与 feed-forward Gaussians**: 虽然不是前向模型，但 conditional slicing 机制很适合作为高维前向 Gaussian 模型的渲染后端。

### 战略权衡

- 优点: 真正统一 time + angle，且保持现有 pipeline 兼容
- 代价: 表示更高维，优化难度和参数解释成本上升

---

## Part III / Technical Deep Dive

### Pipeline

```text
scene elements as 7D Gaussians
-> condition on time and viewing direction
-> conditional slicing to 3D Gaussians
-> render with existing 3DGS pipeline
-> unified dynamic and view-dependent rendering
```

### 关键模块

#### 1. 7D Gaussian Primitive

把位置、时间和视角方向统一到单个 Gaussian primitive 中，是整篇最核心的表示创新。

#### 2. Conditional Slicing

给定时间和视角后，把 7D Gaussian 切成当前条件下的 3D Gaussian。  
这一步保证了高维表示与高效渲染可以兼得。

#### 3. Joint Optimization

因为 unified representation 最终仍能回落到 3DGS 管线，作者得以复用已有优化框架，而不必完全另起炉灶。

### 关键实验信号

- 论文强调 challenging dynamic scenes with complex view-dependent effects
- 相比 prior methods，收益来自“统一建模”而不是只提升某一维表达
- 401 FPS 说明它不是只在 paper idea 层统一，而是真正保持了实时性

### 少量关键数字

- 最多约 `+7.36 dB PSNR`
- 约 `401 FPS`

### 局限、风险、可迁移点

- **局限**: 更高维表示会增加训练和存储复杂度
- **风险**: 条件切片若近似不足，可能损伤某些复杂角度或时间局部现象
- **可迁移点**: high-dimensional Gaussian primitive、conditional slicing、unified appearance-dynamics modeling 都非常值得迁移

### 实现约束

- 需要处理 view-dependent dynamic scenes
- 高度依赖切片机制与 3DGS pipeline 兼容性
- 更偏 foundation representation

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_7DGS_Unified_Spatial_Temporal_Angular_Gaussian_Splatting.pdf]]
