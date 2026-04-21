---
title: "Swift4D: Adaptive divide-and-conquer Gaussian Splatting for compact and efficient reconstruction of dynamic scene"
venue: ICLR
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - divide-and-conquer
  - static-dynamic-decomposition
  - hash-deformation
  - compact-representation
  - status/analyzed
core_operator: 先在 canonical space 中可学习地区分 static 与 dynamic primitives，只为动态部分分配 multi-resolution 4D hash deformation，而静态部分直接保留 3D 表示，从而以 divide-and-conquer 方式压缩动态 4DGS 的训练和存储开销。
primary_logic: |
  先通过可学习分解参数把场景 primitives 划分为 static 和 dynamic 两部分，
  再仅对 dynamic primitives 施加基于 multi-resolution 4D hash mapper 的时间形变，
  最后将 static 与 deformed dynamic primitives 混合渲染，
  从而在保持质量的同时显著降低动态场景重建的训练时间和存储冗余。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICLR_2025/2025_Swift4D_Adaptive_divide_and_conquer_Gaussian_Splatting_for_compact_and_efficient_reconstruction_of_dynamic_scene.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:21
updated: 2026-04-18T16:21
---

# Swift4D: Adaptive divide-and-conquer Gaussian Splatting for compact and efficient reconstruction of dynamic scene

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2503.12307](https://arxiv.org/abs/2503.12307) · [Code](https://github.com/WuJH2001/swift4d)
> - **Summary**: Swift4D 的贡献非常务实: 不是所有高斯都需要被 4D 化。它把动态场景拆成 static primitives 和 dynamic primitives，只给后者分配 deformation 能力，于是把 4DGS 最昂贵的部分限制在真正需要运动建模的区域里。
> - **Key Performance**:
>   - 论文宣称在真实数据集上实现 SOTA 画质，同时训练速度比先前 SOTA 快约 `20x`。
>   - 存储最小可到约 `30 MB`，代表图中标准版本约 `125 FPS`，lite 版本约 `215 FPS`。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

很多动态 4DGS 方法的隐含假设是: 一旦场景是动态的，几乎所有 primitives 都得支持时间形变。  
但现实中大多数区域其实是静态的。  
这带来几个问题:

- 训练浪费大量时间在静态部分
- 存储冗余明显
- deformation network 也会被无意义动态拖累

Swift4D 要解决的是:

**能不能只把 4D 动态建模预算用在真正会动的 primitives 上。**

### 核心能力定义

- **输入**: dynamic scene observations
- **输出**: 更紧凑、更高效的 dynamic Gaussian representation
- **强项**: 训练效率、存储压缩、静动态显式分工
- **弱项**: 若动态区域本身占比很高，收益会下降

### 真正的挑战来源

- 4D deformation 若全场启用，成本会迅速膨胀
- 场景里静态部分往往远多于动态部分
- 直接删除动态或粗暴分割又会损伤质量

### 边界条件

- 方法假设场景中 static primitives 占主导
- 更适合一般动态场景，不专门处理高度全动态拓扑变化
- 仍然是 per-scene optimization，而不是通用 feed-forward inference

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

Swift4D 的核心哲学是:

**动态重建应该采用 selective 4D modeling，而不是 all-in 4D modeling。**

这和很多把所有高斯都送进 deformation network 的方法不同。  
作者先做场景内部分工，再决定哪些 primitive 值得拥有时间维。

### The "Aha!" Moment

真正的 aha 是:

**不是“先有一个统一 deformation，再让它决定哪里变化”，而是“先学会谁是 dynamic，再给它 deformation 能力”。**

这一步把动态建模从“后验补偿”变成“前置资源分配”。  
结果是:

- static 区域直接保留 3D 表示
- dynamic 区域才走 4D hash deformation

因此系统复杂度和存储都被压到了更合理的位置。

### 为什么这个设计有效

因为场景中绝大多数显式粒子并不需要时间自由度。  
只给 dynamic primitives 分配 multi-resolution 4D hash mapper，本质是在压缩时序自由度，而不是压缩整个场景分辨率。

### 对我当前方向的价值

这篇对你当前方向的价值很高，因为它提供了一个非常通用的工程原则:

**把 4D 能力稀疏地施加到最需要它的部分。**

这对 4D 编辑、在线更新、场景压缩都很关键。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 编辑系统往往也不希望全场都变，Swift4D 的 static/dynamic selective modeling 很适合作为编辑前的场景筛分层。
- **与 3DGS_Editing**: 3D 编辑只在空间上选区域，Swift4D 进一步告诉你 4D 里还应先区分“谁需要时间自由度”。
- **与 feed-forward Gaussians**: 这个框架很适合作为 feed-forward 4DGS 的结构先验，让前向模型只预测 dynamic subset 与其 deformation。

### 战略权衡

- 优点: 训练快、存储小、思路清晰
- 代价: 动态分类若出错，会让部分运动区域欠拟合

---

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene
-> learnable primitive decomposition into static and dynamic
-> 4D hash deformation only for dynamic primitives
-> direct 3D rendering for static primitives
-> merge static and dynamic outputs
-> compact dynamic scene reconstruction
```

### 关键模块

#### 1. Learnable Static / Dynamic Decomposition

作者不是用外部标注硬分，而是引入可学习参数去识别哪些 primitives 属于动态部分。

#### 2. Multi-Resolution 4D Hash Mapper

动态部分通过 compact 4D hash deformation 从 canonical space 映射到各个时刻。  
这比大 MLP deformation 更高效。

#### 3. Divide-And-Conquer Rendering

渲染时把 static 与 dynamic primitives 混合。  
系统因此保留 3DGS 的高效静态渲染优势，同时给动态体必要的时间表达。

### 关键实验信号

- 论文把质量、时间、存储一起摆到主结果图里，说明作者优化的是系统整体 Pareto front
- “20x training speedup + 30MB storage” 是其最重要的能力信号
- 代表图给出 standard/lite 两种版本，强调框架弹性

### 少量关键数字

- 训练约快 `20x`
- 最小存储约 `30 MB`
- 代表性渲染速度约 `125 FPS`，lite 版本约 `215 FPS`

### 局限、风险、可迁移点

- **局限**: 全局大面积动态场景下，divide-and-conquer 优势会减弱
- **风险**: 静动态分类错配会直接影响细节恢复
- **可迁移点**: selective 4D modeling、dynamic-only hash deformation、storage-aware scene decomposition 都很值得迁移

### 实现约束

- 假设静态区域占比较大
- 依赖可学习 static/dynamic decomposition
- 偏效率与紧凑性优先

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/ICLR_2025/2025_Swift4D_Adaptive_divide_and_conquer_Gaussian_Splatting_for_compact_and_efficient_reconstruction_of_dynamic_scene.pdf]]
