---
title: "FLEG: Feed-Forward Language Embedded Gaussian Splatting from Any Views via Compact Semantic Representation"
venue: arXiv
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - feed-forward
  - semantic-reconstruction
  - language-embedded-gaussians
  - pose-free
  - multi-view
  - compact-semantic-representation
  - open-vocabulary
  - status/analyzed
core_operator: 用几何-语义双分支蒸馏在无 3D 标注下训练任意视角输入的 feed-forward 语言高斯重建，并用 DGLE 把语义从稠密几何高斯中解耦成少量 semantic Gaussians，显著压缩语义存储。
primary_logic: |
  输入任意数量、无位姿无标定的多视图图像，
  先通过 VGGT 风格主干恢复深度与相机并预测标准高斯，
  训练时用 novel-view-based 几何蒸馏与 CLIP + instance contrast 语义蒸馏共同监督，
  再通过 DGLE 将像素级语言嵌入聚合到更稀疏的 semantic Gaussians，
  最终得到同时支持 novel view synthesis、open-vocabulary query 与 3D editing 的 language-embedded GS。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/arXiv_2025/2025_FLEG_Feed_Forward_Language_Embedded_Gaussian_Splatting_from_Any_Views.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-10T18:24
updated: 2026-04-10T18:24
---

# FLEG: Feed-Forward Language Embedded Gaussian Splatting from Any Views via Compact Semantic Representation

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://fangzhou2000.github.io/projects/fleg) | [arXiv 2512.17541](https://arxiv.org/abs/2512.17541)
> - **Summary**: FLEG 把“前向重建 3DGS”和“前向构建语言语义场”真正合到了一起，而且关键不是简单给每个 Gaussian 挂一个 embedding，而是发现语义本身比几何稀疏得多，于是设计出独立、稀疏、可压缩的 semantic Gaussian 表示。
> - **Key Performance**:
>   - 相比像素对齐的 per-Gaussian embedding，FLEG 只用 `73.89K` 个语言嵌入、约 `18.04MB` 存储，而对照方案需要 `2.012M` 嵌入、约 `491.3MB`，嵌入数低于 `5%`。
>   - 在 `2-view` 的 `ScanNet++ / ScanNet` 上，FLEG 分别达到 `mIoU 46.61 / 48.01` 与 `PSNR 28.14 / 25.05`，同时超过已有 feed-forward 语义高斯方法与普通 2D distillation 基线。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

FLEG 的核心问题非常贴近你当前 idea：

**能不能从任意数量、无位姿、无标定的多视图图像里，一次前向就重建出既可渲染、又带开放词汇语义的 3D Gaussian scene？**

这个问题之所以难，是因为它同时踩中了两个痛点：

- feed-forward 重建通常缺少大规模 3D 几何与语义真值
- 现有语言高斯方法大多给每个 Gaussian 挂一个 language embedding，语义冗余非常严重

### 核心能力定义

- **输入**：任意数量的无位姿多视图图像
- **输出**：同时含几何、颜色和语言语义的 language-embedded Gaussians
- **支持任务**：novel view synthesis、open-vocabulary query segmentation、3D editing
- **特别强调**：一个模型同时兼容 sparse view 与 dense view，而不是固定 2-view 或依赖相机参数

### 真正的挑战来源

- 没有 3D GT、没有 semantic GT、甚至没有相机参数
- 只在输入视角上蒸馏容易过拟合 observed views，导致新视角一致性差
- 若语义和几何使用同等密度表示，存储开销会随着输入视角线性膨胀

### 边界条件

- 论文主要验证静态室内多视图重建与开放词汇查询，不涉及动态时序场景
- 语义能力建立在 CLIP 蒸馏之上，因此更偏开放词汇对齐而非细粒度实例推理

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

FLEG 真正重要的地方，是把问题拆成两个不同密度的子问题：

- **几何/外观** 需要高密度高斯来支撑渲染
- **语义** 并不需要和几何一样稠密

作者由此得到一个非常强的设计原则：

**不要再默认“每个 Gaussian 都必须有一份语言嵌入”。**

### The "Aha!" Moment

论文最值得记住的 aha 是：

**语义场比几何场稀疏得多，因此语言表示应该从几何高斯中解耦出来，单独压缩成更少的 semantic Gaussians。**

这一步改变了什么？

- 传统做法把语言嵌入密集绑定到每个 Gaussian，导致存储与冗余一起膨胀
- FLEG 先用标准高斯负责几何与外观，再让 DGLE 把语义聚合到更大体素下的 semantic Gaussians
- 这样语义表示密度终于和语义本身的复杂度匹配，而不是被几何采样密度绑架

### 为什么这个设计有效？

- 语义蒸馏不再直接依赖 3D semantic GT，而是通过 `CLIP feature alignment + instance-guided contrastive learning` 从 2D 提升到 3D
- 几何蒸馏使用 `VGGT` 伪标签，同时引入 novel-view-based distillation，不只盯着输入视角学，从训练层面缓解多视图过拟合
- semantic Gaussians 使用更大 voxel 聚合，说明模型明确把“语义覆盖区域”视为高层抽象，而不是像素级精细纹理

### 战略权衡

- **优点**：真正适合做大规模语义 3DGS 资产；输入灵活；语义存储极省
- **局限**：当前语义仍以 CLIP 蒸馏为主，细粒度边界与复杂关系推理还不是重点

## Part III / Technical Deep Dive

### Pipeline

```text
arbitrary unposed multi-view images
-> VGGT-style transformer backbone predicts depths and cameras
-> Gaussian head predicts standard Gaussians and pixel-wise language embeddings
-> voxelization builds standard geometric Gaussians
-> geometry branch: VGGT novel-view distillation + photometric supervision
-> semantic branch: SAM2 masks + CLIP alignment + instance-guided contrastive learning
-> DGLE aggregates sparse semantic Gaussians
-> final language-embedded GS supports rendering, query, and editing
```

### 关键模块

#### 1. 几何-语义双分支蒸馏

几何分支依赖冻结 `VGGT` 提供深度与相机伪标签，并通过“选择覆盖充分的新视角”来做 novel-view distillation；语义分支则用 `SAM2` 给实例掩码、`CLIP` 给语言对齐特征，再用 contrastive loss 拉开不同实例之间的语义边界。

#### 2. Novel-View-Based Distillation

这是论文里一个很实用的训练细节。作者不是只对输入视角做蒸馏，而是先检查 context views 对其他图像的覆盖率，只选那些“足够可见”的 novel views 做监督。这样做的结果是：

- 既保留了无相机参数训练的稳定性
- 又显式逼着模型在新视角上保持一致性

#### 3. DGLE: Decoupled Gaussian Language Embedding

DGLE 先构建标准高斯 `G_voxel` 负责几何和颜色，再在更大体素上聚合出 `G_sem` 负责语义。语义高斯去掉 SH 颜色，只保留：

- 位置
- 尺度
- 旋转
- 透明度
- 语言嵌入

其中尺度通过协方差 moment matching 聚合，再做 isotropic 近似。这说明作者关心的是“语义能覆盖哪个区域”，而不是让语义表示追随几何的各向异性细节。

### 关键实验信号

- 在 `2 / 4 / 8 / 16` views 上，FLEG 在 `ScanNet++` 与 `ScanNet` 的 `mIoU / mAcc / PSNR / SSIM / LPIPS` 上整体都领先已有 feed-forward 语义高斯方法
- 在 `32-view` 密集设置下，FLEG 仍显著优于 `LangSplat` 与 `LangScene-X`，例如在 `ScanNet` 上达到 `mIoU 60.97`、`mAcc 91.67`
- ablation 显示：移除 instance-guided contrastive 后，`mIoU` 从 `46.87` 掉到 `40.22`；移除 novel-view distillation 后，`PSNR` 从 `26.66` 掉到 `24.77`
- 去掉几何蒸馏后训练几乎崩掉，`PSNR` 仅 `14.38`，说明该框架本质上高度依赖双分支蒸馏的协同

### 对当前 idea 的启发

FLEG 几乎就是你当前“semantic feed-forward editing”方向最直接的参考之一：

- 它证明了 **feed-forward semantic GS** 可以不靠 per-scene optimization
- `DGLE` 提供了一个很强的答案：语义表示不必与几何一一绑定，完全可以单独稀疏化
- `novel-view distillation + instance contrast` 这组训练策略，很适合给未来的语义接地、语义定位或语义编辑模块做 teacher supervision

### 实现约束

- 主干规模约 `1B` 参数
- 训练使用 `8 x NVIDIA H200`，约 `2` 天
- 训练数据为 `ScanNet / ScanNet++`

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/arXiv_2025/2025_FLEG_Feed_Forward_Language_Embedded_Gaussian_Splatting_from_Any_Views.pdf]]
