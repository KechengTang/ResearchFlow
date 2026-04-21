---
title: "Identity-aware Language Gaussian Splatting for Open-vocabulary 3D Semantic Segmentation"
venue: ICCV
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - language-embedded-gaussians
  - identity-encoding
  - multi-view-identity
  - semantic-segmentation
  - status/analyzed
core_operator: 给每个 Gaussian 同时学习 language embedding 和 identity embedding，用身份一致性约束把跨视角语义对齐，再通过 progressive mask expanding 把查询结果从单点相似度扩成边界更准确的目标区域。
primary_logic: |
  输入 3DGS 场景、多视图图像与近似 identity masks，先让每个 Gaussian 学习语言与身份双重表示，并通过 identity-aware semantic consistency loss 拉近同一对象、拉远不同对象，再在推理时从高相似 seed segment 出发做 progressive mask expanding，最终输出跨视角一致的开放词汇 3D 语义分割。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_Identity_aware_Language_Gaussian_Splatting_for_Open_vocabulary_3D_Semantic_Segmentation.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:23
updated: 2026-04-20T19:23
---

# Identity-aware Language Gaussian Splatting for Open-vocabulary 3D Semantic Segmentation

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Jang_Identity-aware_Language_Gaussian_Splatting_for_Open-vocabulary_3D_Semantic_Segmentation_ICCV_2025_paper.pdf) / [GitHub](https://github.com/DCVL-3D/ILGS)
> - **Summary**: 这篇论文抓住了一个很常被忽略的根因: 开放词汇 3D 分割错，不一定是文本嵌入弱，而是同一物体在不同视角下拿到的语言 embedding 根本不一致。它因此先对齐 identity，再做语言分割。
> - **Key Performance**:
>   - 在 LERF 上达到 `80.5 mIoU / 76.0 mBIoU`，显著高于 Gaussian Grouping 的 `72.8 / 67.6`。
>   - 在 3D-OVS 上达到 `94.4 mIoU`，高于 GOI 的 `90.6`；同时 novel-view 渲染保持 `26.108 PSNR / 0.868 SSIM / 0.210 LPIPS`。

## Part I / The "Skill" Signature

### 它到底在解决什么问题?
ILGS 瞄准的是开放词汇 3D 语义分割里一个很本质的错位:

**同一个对象在不同视角下经常会被赋予不同的语言 embedding，于是模型最后分割出来的是“词相似的碎片”，不是“同一个物体”。**

这会带来两个直接后果:

- 跨视角查询不稳定，同一文本提示在不同视角下激活区域不一致。
- 固定阈值生成 mask 容易在边界处漏掉目标，或者把相邻区域一起吞进去。

因此论文真正想做的是:

- **能做**: 跨视角一致的开放词汇 3D 语义分割。
- **额外目标**: 边界更完整，且不牺牲 novel-view 渲染质量。
- **不主打**: 动态场景、实例级交互式编辑。

### 输入 / 输出接口

- **输入**: 3DGS、多视图图像、CLIP 特征、近似 coherent identity masks。
- **输出**: 每个 Gaussian 上的语言和身份表示，以及文本查询对应的 3D 语义 mask。

### 边界条件

- 身份监督来自 zero-shot tracker 近似标签，不是完美 GT。
- progressive mask expanding 依赖 segment 质量；如果初始分段过差，边界收益会受限。

## Part II / High-Dimensional Insight

### 方法真正新的地方

ILGS 的思路不是继续强化 language feature 本身，而是引入一个额外问题:

**先问“这是不是同一个对象”，再问“它像不像这个词”。**

于是它在每个 Gaussian 上并行学习两种表示:

- language embedding: 负责和文本对齐。
- identity embedding: 负责把同一对象在不同视角下拉到同一簇里。

### The "Aha!" Moment

真正的 aha 在于:

**开放词汇 3D 分割的核心瓶颈，不是词语和外观对齐得不够，而是同一对象在多视角下缺少稳定身份。**

如果身份不稳:

- 语言 embedding 就会在不同视角间漂移。
- 固定阈值得到的 mask 边界也会忽左忽右。

ILGS 的做法因此是两段式:

- 训练时，用 identity-aware semantic consistency loss 把同身份对象的语言表示拉近。
- 推理时，不再用死阈值直接截语言热图，而是从最像查询词的 seed segment 出发，借助 identity feature 向邻域扩展。

这条链路非常清楚:

`身份一致 -> 语言 embedding 不乱漂 -> seed segment 更可信 -> 渐进扩张时边界更稳`

### 为什么这个设计有效?

- identity embedding 不是替代语言，而是给语言对齐提供跨视角锚点。
- outlier filtering 把 noisy tracker 监督中的明显错配先滤掉，避免错误身份拖垮语义场。
- progressive mask expanding 用局部 segment 关系替代全局固定阈值，更适合处理边界细节。

### 策略权衡

- **优点**: 对跨视角一致性和边界都有效；与现有 language-GS 系统兼容度高。
- **代价**: 依赖外部 tracker 质量；依赖已有 3DGS 与场景分段质量。

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS + multi-view images
-> extract CLIP features and approximate identity masks
-> learn per-Gaussian language embedding + identity embedding
-> identity-aware semantic consistency loss + outlier filtering
-> render language / identity feature maps
-> text query selects seed segment
-> progressive mask expanding over neighboring segments
-> open-vocabulary 3D semantic segmentation
```

### 关键信号

- 论文的关键不在更复杂的 backbone，而在“身份约束”这个额外监督维度。
- 它用 mBIoU 明确衡量边界质量，说明作者不是只追求整体 IoU，而是把边界错位视作核心问题。
- ablation 也说明三件事缺一不可: identity loss、mask expanding、outlier filtering 叠加后性能才跳到最好。

### 少量关键数字

- LERF: `80.5 mIoU / 76.0 mBIoU`，显著高于 Gaussian Grouping 的 `72.8 / 67.6`。
- 3D-OVS: `94.4 mIoU`，高于 GOI 的 `90.6`。
- 完整版 ablation 从最弱配置的 `71.4 / 61.7` 提升到 `80.5 / 76.0`，说明三项设计是叠加收益。

### 实现约束

- 论文使用近似 coherent identity labels 监督身份分支。
- novel-view 渲染质量基本不受损，说明身份与语言分支没有明显破坏 3DGS 外观建模。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/ICCV_2025/2025_Identity_aware_Language_Gaussian_Splatting_for_Open_vocabulary_3D_Semantic_Segmentation.pdf]]
