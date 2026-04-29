---
title: "FPGS: Feed-Forward Semantic-aware Photorealistic Style Transfer of Large-Scale Gaussian Splatting"
venue: IJCV
year: 2026
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - feed-forward
  - stylization
  - photorealistic-style-transfer
  - appearance-editing
  - large-scale-scene
  - semantic-correspondence
  - multi-reference-stylization
  - adain-style-injection
  - view-consistency
core_operator: 用风格分解的三维语义场承接 AdaIN 前馈风格化链路，并结合语义对齐与局部 AdaIN，在多参考下对显式 Gaussian 进行一次前馈换装后再实时光栅渲染。
primary_logic: |
  先在目标场景照片上训练可分风格的内容场与语义辅助场，
  再以风格字典与 style attention 在语义对应下为多张参考检索局部风格码，
  最后在内容特征上做一次前馈 AdaIN（含迭代 refinement）后直接得到可渲染的 stylized Gaussians，
  从而在不做每风格优化的前提下兼顾多视角一致性、多参考语义匹配与大规模场景可扩展。
pdf_ref: paperPDFs/3DGS_Editing/IJCV_2026/2026_FPGS_Feed_Forward_Semantic_aware_Photorealistic_Style_Transfer_of_Large_Scale_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-29T12:00
updated: 2026-04-29T12:00
---

# FPGS: Feed-Forward Semantic-aware Photorealistic Style Transfer of Large-Scale Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://kim-geonu.github.io/FPGS/) · [arXiv 2503.09635](https://arxiv.org/abs/2503.09635)
> - **Summary**: FPGS 面向大规模场景的 **写实（photorealistic）前馈风格迁移**：用显式 3D Gaussian 表示场景，先做内容/语义场学习与风格字典检索，再在 3D 特征上做一次前馈 AdaIN 语义风格化，从而在保持多视角一致与实时渲染的同时支持多参考语义匹配；为原会议版 FPRF（NeRF）的期刊扩展版本。
> - **Key Performance**: 论文报告称训练阶段约可比若干 NeRF 前馈/优化式 PST baseline 更高效（例如相对部分方法约 24 分钟量级的训练叙事），并实现实时渲染静态与动态大规模场景的风格化可视化；最终以项目页 / 正式出版 PDF 核定为准。

## Part I：问题与挑战

### 要解决什么？

先前 3D 场景写实风格迁移多依赖 NeRF、逐风格优化或小场景设定，难以同时满足：**前馈应用、多参考语义控制、大规模场景、以及 Gaussian 的实时 splatting**。FPGS 将 stylization 与渲染解耦：先完成整场景前馈风格化，再实时渲染。

### 边界

- 定位是 **appearance / PST**，不是通用几何编辑或文本驱动编辑。
- 与纯 optimization-based 的 3DGS 风格方法（如部分 scaling / per-style 优化路线）在效率—质量权衡上不同，适合作为 **feed-forward + 语义多参考** 对照。

## Part II：方法与洞察

### 核心结构

- **风格分解的 3D 特征场**：内容场编码几何与待解码内容特征；语义场帮助局部语义—风格对应。
- **风格字典 + style attention**：对多参考做聚类压缩，检索与场景语义场对齐的局部风格表示。
- **馈送 AdaIN**：在光栅化前操纵解码颜色统计，使风格应用为前馈；作者还描述 **MLP VGGNet** 与 **迭代风格迁移** 以提升局部质量。

### 核心直觉

**把“风格化”发生在 3D 显式表示上并一次完成，而不是逐视角隐式查询**，才能同时利用 3DGS 的实时渲染与多视角一致性。

## Part III：证据与局限

### 应用与展示

作者展示大规模静/动场景、多参考风格、涂鸦式控制与 4D 风格化等；复现与细节以官方补充材料与代码发布为准。

### 局限与审阅注意

- 训练仍是一个独立阶段；与“零训练、纯推理”的跨场景前馈方法（如部分 single-forward 大模型路线）要区分设定。
- 期刊正式卷期与 DOI 建议以 IJCV 出版社页面为准；本笔记以 arXiv 与项目页为入口。
