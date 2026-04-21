---
title: "OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding"
venue: NeurIPS
year: 2024
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - point-level-understanding
  - open-vocabulary
  - instance-feature-learning
  - two-level-codebook
  - instance-level-association
  - semantic-segmentation
  - status/analyzed
core_operator: 先用无跨帧关联的 SAM boolean masks 学出 3D 一致的 instance features，再用 coarse-to-fine 两级 codebook 把实例特征离散化，最后用实例级 3D-2D 匹配把无损 CLIP 特征挂到 3D Gaussian 上。
primary_logic: |
  输入多视图图像与 3DGS 场景；
  先从 SAM mask 监督下学习 3D 一致、类内紧凑类间分离的 6 维 instance feature；
  再用带坐标的 coarse codebook 加 feature-only 的 fine codebook 做实例离散化；
  最后通过 IoU 与 feature distance 联合匹配，把 2D mask 的 CLIP 特征关联到 3D 实例上，实现点级开放词汇理解。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2024/2024_OpenGaussian_Towards_Point_Level_3D_Gaussian_based_Open_Vocabulary_Understanding.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:06
updated: 2026-04-20T19:06
---

# OpenGaussian: Towards Point-Level 3D Gaussian-based Open Vocabulary Understanding

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [NeurIPS 2024 OpenReview](https://openreview.net/forum?id=3NAEowLh7Q) / [Project Page](https://3d-aigc.github.io/OpenGaussian)
> - **Summary**: OpenGaussian 的核心转向非常明确: 它不满足于“把语言特征渲染回 2D 做 view-consistent 解析”，而是直接追求 3D point-level open-vocabulary understanding，因此把重点放在实例特征的可分性、离散化和无损 CLIP 关联上。
> - **Key Performance**:
>   - 在 LERF 上的 3D object selection 中，平均 `mIoU = 38.36`、`mAcc = 51.43`，显著高于 LangSplat 的 `9.66 / 12.41`。
>   - 在 ScanNet 的 10 类 point-level understanding 上，达到 `38.29 mIoU / 55.19 mAcc`，相比 LangSplat 的 `8.40 / 22.06` 提升明显。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

OpenGaussian 要解决的是：**现有 3DGS 语义方法大多擅长 2D 像素级开放词汇解析，但一到真正的 3D 点级理解就会失真。**

作者把问题拆成两个更本质的卡点：

- **特征表达力不够**：把高维 CLIP 特征压成低维再重建，会损伤 3D 点级判别力。
- **2D-3D 关联不准**：alpha blending 天然不是一一对应，渲染后的 2D feature 好，不等于 Gaussian 本身真的可分。

所以论文的能力定义也很清晰：

- **能做**：文本驱动的 3D object selection、3D point cloud understanding、click-based 3D object selection。
- **不主打**：动态场景、联合几何语义优化、3D detection。

### 输入 / 输出接口

- **输入**：多视图 RGB 图像，SAM 产生的 boolean masks，以及已优化的 3DGS 场景。
- **输出**：每个 Gaussian 上的 point-level open-vocabulary feature，使文本或点击都能直接选出对应 3D 点集。

### 边界条件

- 论文默认几何属性是固定的，只优化实例特征与语言关联。
- 方法依赖 SAM 掩码质量；如果实例在 2D 掩码里就没被稳定覆盖，后续 3D 表征也会受限。
- 作者在结论里明确承认尚未处理动态因素，也没有把方法扩展到 3D detection。

## Part II / High-Dimensional Insight

### 方法真正新的地方

OpenGaussian 没有沿着“继续训练一个更好的 3D 语言场”这条路走，而是先问了一个更基础的问题：

**如果 3D Gaussian 本身还没有清晰的实例边界，那再好的语言特征也挂不稳。**

因此它先做“实例”，再做“语义”：

1. 用 SAM 的 **boolean mask** 学 3D 一致的 instance features，而不是直接蒸馏高维 SAM / CLIP feature。
2. 用 **两级 codebook** 把连续 instance feature 离散成更稳的实例簇。
3. 再把实例与 2D mask 的 CLIP 特征关联起来，从而保留高维、无损的开放词汇能力。

### The "Aha!" Moment

真正的 aha 是：

**点级开放词汇理解的瓶颈不在语言，而在“对象身份”是否已经在 3D 里稳定成形。**

OpenGaussian 的策略是把任务分成两段：

- 第一段只关心“哪些 Gaussian 属于同一实例”，通过类内平滑、类间分离和 codebook 离散化，先把 3D 实例边界做稳。
- 第二段才把 CLIP 特征接进来，而且不是压缩学习，而是通过实例级 3D-2D 关联把高维特征无损贴回去。

这条因果链非常干净：

`实例特征先稳定 -> 离散实例簇更清楚 -> 3D-2D 关联不再依赖逐像素深度测试 -> 高维 CLIP 特征无损挂到实例上 -> 点级开放词汇理解成立`

### 为什么这个设计有效

- 只用 boolean masks 学实例特征，监督更便宜，也避免直接拟合高维 foundation feature。
- 两级 codebook 把“远处不共视物体被挤进同一类”这个单层量化问题拆掉了。
- 关联 CLIP 时使用 **IoU + feature distance** 联合打分，避免了仅靠投影几何或仅靠外观相似带来的错配。

### 策略权衡

- **优点**：更适合点级 3D 理解；高维语言特征无损；点击与文本接口都成立。
- **代价**：流程明显比 LangSplat 一类方法更偏“实例化”而非直接语言场建模；几何固定后，语义与几何不一致的问题仍可能残留。

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view RGB + SAM boolean masks
-> learn 6D per-Gaussian instance features with intra/inter-mask losses
-> coarse-to-fine two-level codebook discretization
-> render single-instance maps
-> match 3D instances with 2D SAM masks by IoU + feature distance
-> associate mask-level CLIP features to 3D instances
-> text query or click query for point-level 3D selection
```

### 关键技术信号

- **3D consistency-preserving instance feature learning**  
  作者没有做跨视角 mask tracking，而是利用 3DGS 的全局一致性，让同一物体在不同视角渲染出的 feature 自动聚拢。

- **two-level codebook**  
  单层 codebook 容量不够时，简单增大 k 反而会把远距离、不共视的点混到同一簇。OpenGaussian 通过 coarse level 引入坐标、fine level 只看 feature，显式解决了这个问题。

- **instance-level 3D-2D association without depth test**  
  这一步很关键。作者没有做逐点深度可见性测试，而是先把 3D 实例渲染成 single-instance map，再和 2D SAM mask 用 IoU 与 feature distance 联合匹配，把 CLIP 特征挂上去。

### 少量关键数字

- 在 LERF 的 3D object selection 上，OpenGaussian 达到 `38.36 mIoU / 51.43 mAcc`，而 LangSplat 只有 `9.66 / 12.41`。
- 在 ScanNet 10 类 point-level understanding 上，达到 `38.29 mIoU / 55.19 mAcc`。
- ablation 显示，两级 codebook 是必要的；仅用单层 `k=64` 时只有 `28.68 mIoU`，而最佳两级配置达到 `38.29 mIoU`。

### 实现约束

- 每个 Gaussian 学的是 `6` 维 instance feature。
- coarse codebook 结合 feature 与 `xyz`，fine codebook 只对 feature 再离散。
- ScanNet 实验中，作者冻结输入点云坐标并关闭 densification，保证输出点云与 GT 点云逐点对齐。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2024/2024_OpenGaussian_Towards_Point_Level_3D_Gaussian_based_Open_Vocabulary_Understanding.pdf]]
