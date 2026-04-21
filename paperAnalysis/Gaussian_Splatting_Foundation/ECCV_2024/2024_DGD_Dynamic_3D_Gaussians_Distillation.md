---
title: "DGD: Dynamic 3D Gaussians Distillation"
venue: ECCV
year: 2024
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 4dgs
  - splatted-features
  - semantic-grounding
  - motion-tracking
  - monocular-video
  - status/analyzed
core_operator: 在 dynamic 3D Gaussians 上联合优化外观与语义属性，把单目动态视频蒸馏成可渲染、可分割、可跟踪的统一动态语义高斯场。
primary_logic: |
  输入单目视频后，先构建基于 dynamic 3D Gaussians 的时变场景表示，
  再联合优化颜色、几何与语义属性，使语义学习反过来影响几何组织，
  最终支持新视角渲染、3D 语义分割以及基于点击或文本提示的密集对象跟踪。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/ECCV_2024/2024_DGD_Dynamic_3D_Gaussians_Distillation.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T15:06
updated: 2026-04-18T15:06
---

# DGD: Dynamic 3D Gaussians Distillation

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2405.19321) / [PDF](https://arxiv.org/pdf/2405.19321.pdf) / [Project Page](https://isaaclabe.github.io/DGD-Website/)
> - **Summary**: DGD 可以看成 Feature 3DGS 在动态场景上的对应物。它不是单纯给动态高斯加语义标签，而是让语义和外观一起参与优化，因此更像一个面向动态语义查询与跟踪的统一 4D 高斯表示。
> - **Key Performance**:
>   - 论文展示了高质量动态语义渲染和密集 3D 目标跟踪。
>   - 交互接口支持点击或文本提示，这对后续编辑定位很有参考意义。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

动态 3DGS 里如果只有颜色和几何，没有稳定语义场，就很难做：

- 对象级查询
- 语义分割
- 动态对象跟踪

DGD 就是在解决这个动态语义表示缺口。

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它的关键不是“蒸馏语义到高斯”，而是：
**让语义优化与几何/外观优化联合进行。**

这样语义不会变成附加通道，而是会反过来影响动态高斯的组织方式。

### The "Aha!" Moment

真正的 aha 是：

**动态语义场如果只是后挂在高斯上，通常不够稳；它必须进入表示优化本身。**

这和你后面做 4DGS 编辑很相关，因为编辑常常依赖能持续跟踪对象身份的语义锚点。

### 权衡与局限

- 优势：语义、几何、外观一起建模
- 局限：主要面向 query / tracking / segmentation，不直接解决高保真编辑

## Part III / Technical Deep Dive

### Pipeline

```text
monocular dynamic video
-> dynamic 3D Gaussians representation
-> joint optimization of appearance and semantics
-> dynamic semantic radiance field
-> novel-view rendering + semantic query + dense 3D tracking
```

### 关键技术点

#### 1. Unified Dynamic Semantic Representation

论文把颜色、几何和语义揉进同一个动态高斯表示，而不是分别训练再对齐。

#### 2. Promptable 3D Tracking

点击或文本提示都能触发语义实体跟踪，这一点对后续做语言驱动编辑定位尤其有价值。

### 对你的价值

DGD 在你的库里适合作为 4DGS 编辑的语义前置基础论文保留，因为它直连：

- 语义 grounding
- 对象持续身份
- 文本/点击驱动的动态对象定位

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/ECCV_2024/2024_DGD_Dynamic_3D_Gaussians_Distillation.pdf]]
