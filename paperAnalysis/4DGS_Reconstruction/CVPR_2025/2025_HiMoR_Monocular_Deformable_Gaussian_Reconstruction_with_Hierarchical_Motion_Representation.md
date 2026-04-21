---
title: "HiMoR: Monocular Deformable Gaussian Reconstruction with Hierarchical Motion Representation"
venue: CVPR
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - monocular-dynamic-scene
  - hierarchical-motion
  - motion-bases
  - tree-structure
  - status/analyzed
core_operator: 用树状 hierarchical motion representation 把动态场景运动分解为 coarse-to-fine 的相对运动，并为不同节点组共享少量 motion bases，在保持时空平滑的同时恢复更细节的单目动态形变。
primary_logic: |
  先在 canonical Gaussian 场中构建层次化 motion tree，
  让浅层节点表示 coarse motion、深层节点表示 fine motion，
  再用共享 motion bases 参数化同一父节点下子节点的相对运动，
  最终通过层次传播和邻近叶节点插值得到每个 Gaussian 的时序形变。
pdf_ref: paperPDFs/4DGS_Reconstruction/CVPR_2025/2025_HiMoR_Monocular_Deformable_Gaussian_Reconstruction_with_Hierarchical_Motion_Representation.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:12
updated: 2026-04-18T16:12
---

# HiMoR: Monocular Deformable Gaussian Reconstruction with Hierarchical Motion Representation

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2504.06210](https://arxiv.org/abs/2504.06210)
> - **Summary**: HiMoR 的核心贡献是把单目动态高斯里的运动表示从“平铺的一组 bases 或 nodes”升级成层次化运动树。它假设真实世界运动天然具有 coarse-to-fine 结构，因此让粗运动保证平滑、细运动补充细节，从而比纯全局 bases 更能抓细节，比大量自由节点又更不容易过拟合。
> - **Key Performance**:
>   - 论文在标准 novel view synthesis 结果上给出 `23.400 PSNR / 0.7621 SSIM / 0.09370 LPIPS` 的代表性结果。
>   - 同时作者强调 perceptual metric 比纯像素指标更能反映 monocular dynamic reconstruction 的真实质量。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

单目动态重建的难点，在于既要从极少观测里恢复复杂运动，又不能把时间信息拟合成训练视角过拟合。  
现有方法常落在两端:

- 全局 motion bases 太少，抓不住细节
- 大量 motion nodes 自由度太高，容易过拟合

HiMoR 的目标是:

**找到一个既能表达细运动，又能保持时空平滑与结构约束的中间运动表示。**

### 核心能力定义

- **输入**: monocular dynamic video
- **输出**: 具备时空一致性的动态 Gaussian reconstruction
- **强项**: 粗细运动分解、单目设定下的平滑与细节兼顾
- **弱项**: 极端拓扑变化、传感器鲁棒性、多模态语义控制

### 真正的挑战来源

- 单目没有多视角一致性约束
- 运动通常具有层次结构，但常见表示没显式编码这种结构
- 评价指标本身也可能误导单目动态重建

### 边界条件

- 方法默认运动具有层次性、平滑性和低秩倾向
- 树状结构与共享 bases 适合连续动态，不专门处理剧烈拓扑断裂
- 仍然是 per-scene optimization 而非 feed-forward 推断

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

HiMoR 的方法哲学非常清晰:

**现实运动往往不是一层平铺的自由变量，而是“粗运动包住细运动”的层级系统。**

所以作者不再把所有 Gaussian 的运动都交给一个统一低秩 bases，也不让海量节点自由插值，而是引入 motion tree。

### The "Aha!" Moment

真正的 aha 是:

**复杂运动可以先拆成 coarse motion 再叠 fine motion，而不是在一个平面参数空间里硬拟合。**

作者通过树状层级让:

- 浅层节点负责大范围、平滑的主运动
- 深层节点负责局部细节
- 同层局部结构再共享少量 motion bases

这相当于同时引入了 **层次先验** 和 **低秩先验**。  
前者抑制粗运动失真，后者防止细运动过拟合。

### 为什么这个设计有效

因为很多真实动态都具有天然的 parent-child 运动关系。  
例如手臂带动手腕、手腕带动手指。  
把这种结构写进 motion representation 后，模型就不需要从稀少单目监督里自己“猜”层次关系。

### 对我当前方向的价值

对你当前方向，这篇论文非常实用，因为它提供了一个高质量 monocular 4DGS operator:

- **hierarchical motion tree**
- **shared motion bases per local group**
- **coarse-to-fine deformation decomposition**

如果你之后做 4D 编辑或 feed-forward monocular 4D，这是很值得蒸馏的结构先验。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 动态编辑往往需要修改能沿主运动稳定传播，HiMoR 的 coarse-to-fine motion hierarchy 很适合作为编辑传播骨架。
- **与 3DGS_Editing**: 3D 编辑一般没有时间层次问题，但其“先粗后细”的控制思想可以迁移到 part-level 或 drag-based control。
- **与 feed-forward Gaussians**: 它不是 feed-forward，但非常适合成为 monocular feed-forward 4DGS 的 teacher structure，因为层次运动比无结构位移更容易泛化。

### 战略权衡

- 优点: 结构化运动、细节与平滑兼顾
- 代价: tree design 与层次优化更复杂，且假设运动分解合理

---

## Part III / Technical Deep Dive

### Pipeline

```text
monocular video
-> canonical Gaussian scene
-> hierarchical motion tree with relative motions
-> shared motion bases for node groups
-> recursive composition to world motion
-> KNN interpolation from leaf nodes to Gaussians
-> dynamic rendering and reconstruction
```

### 关键模块

#### 1. Hierarchical Motion Tree

每个节点表示相对父节点的运动。  
整条树把全局粗运动到局部细运动层层展开。

#### 2. Shared Motion Bases

同一父节点下的子节点共享少量 motion bases，但各自拥有系数。  
这一步保留了低秩建模的简洁，同时避免纯全局 basis 太粗糙。

#### 3. Gaussian Deformation by Leaf Interpolation

最终每个 Gaussian 的形变由周围叶节点的运动插值得到。  
因此运动既带有局部结构，也不会无限自由。

#### 4. Perceptual Evaluation Reframing

作者还专门指出单目动态重建里 PSNR 之类像素指标可能失真，因此引入更可靠的 perceptual metric 作为补充。  
这说明论文不仅改模型，也在改评估逻辑。

### 关键实验信号

- 论文强调同时提升 spatio-temporal smoothness 和 detail restoration
- 相比 SoM 类全局 bases 与 MoSca 类大量节点方法，它主打的是“中间地带”的结构先验
- 结果不仅靠像素指标，还通过感知指标支持其视觉质量主张

### 少量关键数字

- 代表性 quantitative 结果约为 `23.400 PSNR / 0.7621 SSIM / 0.09370 LPIPS`

### 局限、风险、可迁移点

- **局限**: 若运动层次关系本身不明显，树状先验的收益会下降
- **风险**: 层次划分不合适可能造成粗层过强或细层表达不足
- **可迁移点**: hierarchical motion prior、group-shared bases、perceptual evaluation for monocular 4D 都值得迁移到编辑与 feed-forward 研究

### 实现约束

- 单目动态重建设定
- 依赖层次运动树和共享 SE(3) motion bases
- 更重视结构先验而非极简网络

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/CVPR_2025/2025_HiMoR_Monocular_Deformable_Gaussian_Reconstruction_with_Hierarchical_Motion_Representation.pdf]]
