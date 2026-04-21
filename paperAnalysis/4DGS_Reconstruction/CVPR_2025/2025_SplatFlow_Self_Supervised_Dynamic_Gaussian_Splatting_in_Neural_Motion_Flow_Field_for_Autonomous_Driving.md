---
title: "SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field for Autonomous Driving"
venue: CVPR
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - autonomous-driving
  - self-supervised
  - neural-motion-flow-field
  - static-dynamic-decomposition
  - status/analyzed
core_operator: 在 Neural Motion Flow Field 中统一建模 LiDAR 点与 4D Gaussian 的连续时间运动，以自监督方式完成 static/dynamic decomposition，并借助 2D foundation features 提升动态体识别与跨时刻对应。
primary_logic: |
  先以 NMFF 表达 LiDAR 点和 Gaussians 的连续 motion flow，
  再利用该流场把静态背景表示为 3D Gaussians、动态前景表示为 4D Gaussians，
  同时通过时间对应聚合动态 Gaussian 的时序特征并蒸馏 2D foundation features，
  最终在不依赖 tracked 3D bounding boxes 的情况下实现自动驾驶场景重建与 novel-view RGB/depth/flow 合成。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2025/2025_SplatFlow_Self_Supervised_Dynamic_Gaussian_Splatting_in_Neural_Motion_Flow_Field_for_Autonomous_Driving.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:17
updated: 2026-04-18T16:17
---

# SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field for Autonomous Driving

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2411.15482](https://arxiv.org/abs/2411.15482)
> - **Summary**: SplatFlow 的关键价值是把 street-scene 4DGS 从“依赖 3D bbox 的半监督工程系统”推进到更可扩展的自监督方案。它不是只做 static/dynamic 分离，而是把 LiDAR 点与 Gaussians 一起放进同一个 Neural Motion Flow Field 里，让时间对应、前景分离和动态重建共享同一个连续运动场。
> - **Key Performance**:
>   - 论文在 Waymo 和 KITTI 上宣称达到 image reconstruction 与 novel view synthesis 的 SOTA。
>   - 运行时表中，SplatFlow 约为 `40 FPS`（Waymo）和 `44 FPS`（KITTI），在动态 4D 街景方法里仍保持较强实时性。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

自动驾驶动态街景重建有一个很现实的问题:

- 复杂场景里动态体很多
- 依赖 tracked 3D bounding boxes 的方法很难大规模扩展
- 但纯 self-supervised 方法又容易分不清谁在动、怎么动、跨时间怎么对应

SplatFlow 要解决的是:

**在没有昂贵 object-level 标注的情况下，能否仍然稳定地做 street-scene dynamic Gaussian reconstruction。**

### 核心能力定义

- **输入**: 自动驾驶多帧观测与 LiDAR 信号
- **输出**: 可合成 RGB/depth/flow 的 street-scene 4D Gaussian representation
- **强项**: 自监督动态分解、跨时间对应、街景应用
- **弱项**: 依赖多模态传感器、并非通用单目 4DGS

### 真正的挑战来源

- street scene 的动态体类别多、速度差异大
- 若没有 3D bbox，dynamic object identification 很容易失稳
- 只做静动态二分类还不够，还需要可连续查询的 motion correspondence

### 边界条件

- 方法明显面向 autonomous driving
- 依赖 LiDAR 先验和 foundation feature 蒸馏
- 目标是高质量街景重建与视图合成，不是语义编辑

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

SplatFlow 的核心思路是:

**不要把动态分解、时间对应和几何重建拆成多套独立模块，而是把它们统一到连续 motion flow field 里。**

于是动态建模不再只是 per-Gaussian 的位移拟合，而是一个共享时空运动场。

### The "Aha!" Moment

真正的 aha 是:

**如果能同时让 LiDAR points 和 4D Gaussians 共享一个 Neural Motion Flow Field，那么 dynamic object decomposition 与 temporal correspondence 就有了共同的几何支撑。**

这比单纯依赖 2D 检测或 3D bbox 更统一，也比完全无结构的 time-conditioned deformation 更稳定。

### 为什么这个设计有效

NMFF 同时承担三件事:

- 表达动态体的连续运动
- 帮助分离静态背景和动态对象
- 为每个 4D Gaussian 提供跨时间对应关系

在此基础上再蒸馏 2D foundation features，作者进一步强化了动态对象识别，减少仅靠几何自监督带来的歧义。

### 对我当前方向的价值

这篇论文对你当前方向很有价值，因为它说明了:

**street-scene 4DGS 的关键不只是更强的 Gaussian 表示，而是更统一的 motion prior。**

尤其是如果你后续关心动态体分解、运动传播或驾驶场景编辑，这种“共享 motion field”思路非常值得保留。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 一旦要在动态街景里只编辑某个 moving actor，稳定的时序对应是前提；SplatFlow 的 NMFF 正是在做这个底层准备。
- **与 3DGS_Editing**: 3D 编辑通常没有连续时间对应问题，而 SplatFlow 补上的是 4D 情况下编辑对象的轨迹级一致性。
- **与 feed-forward Gaussians**: 虽然不是 feed-forward，但 NMFF + foundation feature 的组合很适合作为未来街景前向 4DGS 的监督模板。

### 战略权衡

- 优点: 去掉了 3D bbox 依赖，统一动态分解和时间对应
- 代价: 系统复杂，且依赖 LiDAR 与 foundation features

---

## Part III / Technical Deep Dive

### Pipeline

```text
driving observations + LiDAR
-> Neural Motion Flow Field
-> static background as 3D Gaussians
-> dynamic foreground as 4D Gaussians
-> temporal correspondence aggregation
-> 2D foundation feature distillation
-> RGB/depth/flow synthesis
```

### 关键模块

#### 1. Neural Motion Flow Field

NMFF 是整个系统的骨架。  
它把点和高斯都放进一个连续运动场里建模，而不是让每个高斯各自学独立轨迹。

#### 2. Static/Dynamic Gaussian Split

背景用 3D Gaussians，前景用 4D Gaussians。  
这种分工既节省表示开销，也更符合街景动态分布。

#### 3. Temporal Feature Aggregation

NMFF 提供跨时间对应后，动态高斯可以聚合时序特征，从而增强跨视角一致性。

#### 4. Foundation Feature Distillation

作者借助 2D foundation models 的特征来改善动态对象识别，说明纯几何自监督仍不够，需要语义/视觉先验补充。

### 关键实验信号

- 论文明确强调不依赖 tracked 3D bounding boxes
- 在 Waymo/KITTI 上同时看 image reconstruction 和 novel view synthesis，说明目标不是单一指标刷分
- 运行时表表明方法仍保留较强实时性

### 少量关键数字

- 推理速度约 `40 FPS`（Waymo）与 `44 FPS`（KITTI）

### 局限、风险、可迁移点

- **局限**: 更偏 street-scene + LiDAR 场景，迁移到普通视频并不直接
- **风险**: dynamic decomposition 若受 foundation features 误导，可能把静态区域误分成动态
- **可迁移点**: unified motion field、temporal correspondence supervision、feature-assisted dynamic decomposition 都值得迁移

### 实现约束

- 自动驾驶多模态设定
- 依赖 LiDAR、motion field 和 2D feature distillation
- 更强调 scalability than manual labeling

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/CVPR_2025/2025_SplatFlow_Self_Supervised_Dynamic_Gaussian_Splatting_in_Neural_Motion_Flow_Field_for_Autonomous_Driving.pdf]]
