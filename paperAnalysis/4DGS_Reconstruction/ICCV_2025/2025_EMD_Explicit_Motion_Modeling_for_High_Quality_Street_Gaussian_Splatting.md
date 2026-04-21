---
title: "EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting"
venue: ICCV
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - street-scene
  - motion-embedding
  - motion-decomposition
  - self-supervised-driving
  - status/analyzed
core_operator: 通过给 Gaussian 引入 learnable motion embeddings 显式编码不同动态体的运动模式，并以 plug-and-play 的 Explicit Motion Decomposition 模块增强 street-scene static/dynamic separation，尤其补足自监督街景 4DGS 中对速度差异建模不足的问题。
primary_logic: |
  先在街景 Gaussian 表示中为动态对象注入 learnable motion embeddings，
  再借助 EMD 模块把不同运动模式显式纳入 scene decomposition，
  同时配合定制训练策略兼容自监督与带监督街景方法，
  最终提升动态街景的重建清晰度、动态体分离质量与 novel-view synthesis 表现。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_EMD_Explicit_Motion_Modeling_for_High_Quality_Street_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:17
updated: 2026-04-18T16:17
---

# EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2411.15582](https://arxiv.org/abs/2411.15582) · [Project Page](https://qingpowuwu.github.io/emd)
> - **Summary**: EMD 的切入点非常准确: 很多 street Gaussian 方法已经会做静动态分解，但它们没有真正理解“不同动态体的运动模式不同”。结果就是 pedestrian、vehicle、slow actor 都被差不多地建模，导致分解不准、重建发糊。EMD 用 learnable motion embeddings 把这层 motion heterogeneity 显式写进 Gaussian 表示。
> - **Key Performance**:
>   - 论文强调在 self-supervised street setting 下达到 SOTA novel view synthesis 效果。
>   - 结果页同时报告 full-image 与 vehicle-region 的 PSNR/SSIM/LPIPS，说明作者特别关注动态对象区域的改善。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

动态街景的难点不只是“哪些东西会动”，还在于 **它们动得不一样**:

- 行人慢
- 车辆快
- 不同对象的轨迹和速度分布差异很大

现有方法虽然会做 static/dynamic split，但多数没有显式 motion modeling，尤其 self-supervised 方法更明显。  
EMD 的目标是:

**让 street Gaussian splatting 真正感知不同动态体的运动模式差异，从而提升分解和重建质量。**

### 核心能力定义

- **输入**: 动态街景训练数据
- **输出**: 动态体分解更准确、重建更清晰的 street Gaussian scene
- **强项**: motion-aware decomposition、street-scene self-supervision、动态体质量提升
- **弱项**: 非街景泛化、通用 4D scene representation

### 真正的挑战来源

- 监督式方法常依赖 3D bbox 或分割，代价高
- 自监督方法虽然更 scalable，但缺少显式 motion prior
- 仅做静动态二分，无法表达不同 actor 的 motion spectrum

### 边界条件

- 方法最适合动态街景和驾驶数据
- 更像 plug-in module，而不是全新 4DGS 框架
- 主要优化 dynamic decomposition 和 render quality

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

EMD 的设计哲学很直接:

**街景动态建模里，最缺的不是再一个更大网络，而是把“运动模式差异”显式表示出来。**

因此作者提出 motion embeddings，让 Gaussian 不只区分 static / dynamic，还携带更细的 motion identity。

### The "Aha!" Moment

真正的 aha 是:

**dynamic scene decomposition 不应该是二值分解，而应该是基于 motion spectrum 的显式分解。**

EMD 通过 motion embeddings 把这一点写进高斯表征后，动态对象的边界和清晰度都会改善，因为模型终于知道“谁和谁虽然都在动，但不是同一种动法”。

### 为什么这个设计有效

在街景中，motion speed 和 motion pattern 是分解动态对象的重要线索。  
如果模型只知道“动/不动”，就会把许多不同对象混成一团。  
EMD 通过可学习 motion embeddings 给 Gaussian 增加了一个额外判别轴，这会直接改善动态分离与 render sharpness。

### 对我当前方向的价值

这篇论文对你当前方向有两个实际价值:

- 它证明 dynamic decomposition 可以不是 binary，而是 embedding-based
- 对后续 4D 编辑而言，这类 motion identity 也可以变成编辑控制信号

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 如果未来要按 actor 的运动模式编辑动态场景，EMD 这种 motion embedding 会比纯 mask 更自然。
- **与 3DGS_Editing**: 3D 编辑通常只要空间分区；4D 编辑则还需要时间上“谁属于哪种运动”的归类，EMD 提供了这条线索。
- **与 feed-forward Gaussians**: 它不是 feed-forward，但 motion embeddings 很适合作为前向模型的中间表示或辅助监督。

### 战略权衡

- 优点: 插件化、针对性强、特别适合自监督街景方法
- 代价: motion embedding 的解释性有限，且街景外的普适性待验证

---

## Part III / Technical Deep Dive

### Pipeline

```text
street-scene Gaussian representation
-> inject learnable motion embeddings into dynamic Gaussians
-> explicit motion decomposition for static/dynamic separation
-> tailored training for self-supervised or supervised variants
-> sharper reconstruction and novel-view synthesis
```

### 关键模块

#### 1. Learnable Motion Embeddings

Gaussian 不再只是一个外观+位置粒子，还携带 motion-aware embedding。  
这让模型有能力区分不同动态体的运动模式。

#### 2. Explicit Motion Decomposition Module

EMD 作为 plug-and-play module，核心任务就是补足街景 4DGS 中“运动建模这条腿”。

#### 3. Tailored Training Strategies

作者没有把方法限制在某一个范式内，而是为自监督和监督式方法都设计了兼容训练策略，增强了模块通用性。

### 关键实验信号

- 论文不只看 full-image 指标，还专门看 vehicle region，说明收益主要体现在动态对象
- 重点宣称 self-supervised setting 下的 SOTA，表明其价值在于补足无标注方案的 motion prior
- Figure 1 的 error map 直接显示动态体变得更锐利

### 少量关键数字

- 论文在 Waymo/KITTI 上同时报告 full-image 与 vehicle 区域的 PSNR/SSIM/LPIPS

### 局限、风险、可迁移点

- **局限**: 场景依赖较强，主要针对 street-scene motion heterogeneity
- **风险**: 若 embedding 学到的是数据集偏差而不是稳定 motion pattern，泛化会受限
- **可迁移点**: motion identity embedding、non-binary dynamic decomposition、region-aware dynamic evaluation 都值得迁移

### 实现约束

- 动态街景重建
- 更适合作为现有方法的增强模块
- 重点服务 dynamic actor quality

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_EMD_Explicit_Motion_Modeling_for_High_Quality_Street_Gaussian_Splatting.pdf]]
