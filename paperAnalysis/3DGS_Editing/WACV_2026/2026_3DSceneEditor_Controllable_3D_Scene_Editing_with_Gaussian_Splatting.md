---
title: "3DSceneEditor: Controllable 3D Scene Editing with Gaussian Splatting"
venue: WACV
year: 2026
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - scene-editing
  - 3d-only-editing
  - open-vocabulary-grounding
  - interactive-editing
  - object-grounding
  - localized-editing
  - prompt-driven-editing
  - status/analyzed
core_operator: 用 3D instance segmentation 给 Gaussians 赋语义，再通过 GPT-4V 关键词解析、视角相关空间关系推理和 CLIP 语言-对象对齐做 3D ROI grounding，最后直接在 ROI 内执行 Gaussian 级编辑操作。
primary_logic: |
  输入已重建的 3D Gaussian scene 与文本编辑指令，
  先用预训练 3D instance segmentation 为每个 Gaussian 赋标签，
  再解析 prompt 中的目标对象、动作与空间关系，通过 view-based grounding 和 CLIP 对齐确定 ROI，
  最后直接在目标高斯上执行删除、增添、换色、移动、替换等操作，
  并用 KNN 优化修正边界噪声，得到交互式可控编辑结果。
pdf_ref: paperPDFs/3DGS_Editing/WACV_2026/2026_3DSceneEditor_Controllable_3D_Scene_Editing_with_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-10T19:43
updated: 2026-04-10T19:43
---

# 3DSceneEditor: Controllable 3D Scene Editing with Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [WACV PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Yan_3DSceneEditor_Controllable_3D_Scene_Editing_with_Gaussian_Splatting_WACV_2026_paper.pdf)
> - **Summary**: 3DSceneEditor 的主张很直接: 把复杂场景编辑从“先渲染 2D、再分割/投影/扩散回 3D”的多步流程，改成“先在 3D 高斯里找准对象，再直接做高斯级操作”的 3D-only 编辑范式。
> - **Key Performance**:
>   - 在量化对比中达到 `TIS 23.28% / IIS 96.17%`，同时优于对比方法。
>   - 初次编辑约 `2-5 min`、二次编辑低于 `1 min`，VRAM 约 `9.4GB / 9.1GB`；用户偏好中 `Prompt Alignment 92.86%`、`Editing Quality 90.48%`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文真正想解决的是：

**如何在复杂 3D 场景里做精确、可控、交互式的对象级编辑，而不再依赖 2D diffusion 与 2D-3D 投影链路？**

此前很多 Gaussian 编辑方法虽然可用，但常见问题也很集中：

- 编辑依赖 InstructPix2Pix 一类 2D diffusion，输出受分辨率和图像先验限制
- 要先在多视角图像中检测/分割，再逐帧投回 3D，流程长且容易误差积累
- 对复杂 indoor/outdoor scene 的对象级定位仍不够稳定

### 核心能力定义

- **输入**：一个已重建 3D Gaussian scene 和文本编辑指令
- **输出**：编辑后的 3D Gaussian scene
- **支持操作**：object removal、addition、recoloring、repositioning、replacement
- **擅长**：复杂布局中的局部对象控制、重复物体间的目标定位

### 真正的挑战来源

- 同类目标在复杂场景中常重复出现，仅靠类别名很难唯一定位
- 纯 2D 编辑难以保持多视角一致与场景风格一致
- 直接移动或替换 3D Gaussians 会引入边界噪声与空间伪影

### 边界条件

- 论文默认已有高质量的 3D Gaussian scene
- 物理合理性不是当前重点；大范围物体移动仍受限制，尤其是会影响大量 ray-space 关系的情况

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

3DSceneEditor 的设计哲学可以概括成一句话：

**先在 3D 里把“要改哪里”搞准，再在 3D 里把“怎么改”直接做掉。**

这和 diffusion-based 编辑的范式有根本差异。后者通常是：

- 先把 3D scene 渲染成 2D
- 再在 2D 做编辑
- 再试图把编辑结果回写到 3D

而 3DSceneEditor 是先建立 3D semantic understanding，再把编辑限制在 ROI 内直接操作高斯。

### The "Aha!" Moment

真正的 aha 是：

**复杂场景编辑的瓶颈不只是生成能力，而是 3D 目标 grounding。只要 3D grounding 足够准，很多对象级编辑根本不需要 diffusion。**

作者因此把系统分成三层：

- 3D instance segmentation：先给每个 Gaussian 语义标签
- open-vocabulary grounding：再根据 prompt 的类别、空间关系和颜色/动作词定位目标 ROI
- direct Gaussian editing：最后只在 ROI 内做几何或颜色修改

其中最有意思的是第二层。它并非只做文本相似度匹配，而是还要解释：

- 左边 / 右边 / 中间 / 靠近等空间关系
- 同类对象之间谁才是 prompt 指向的那个

### 为什么这个设计有效？

- 把视角相关的空间关系先投影到一个虚拟 2D egocentric 平面中，能更稳定地区分同类对象
- CLIP 只负责最后的语言-对象对齐，不必单独承担全部定位逻辑
- 编辑只发生在 3D ROI 内，因此风格保持和多视角一致性天然更稳

### 战略权衡

- **优点**：定位准、操作直、交互速度快，特别适合对象级局部编辑
- **局限**：更像 controllable 3D manipulation framework，而不是通用外观生成器；风格化与物理合理性仍有限

## Part III / Technical Deep Dive

### Pipeline

```text
3D Gaussian scene + text instruction
-> pre-trained 3D instance segmentation assigns semantic labels to Gaussians
-> GPT-4V extracts object/action/relation/color keywords
-> view-based spatial relation reasoning filters candidate objects
-> CLIP computes language-object correlation and selects ROI
-> direct Gaussian editing in ROI
-> KNN-based optimization cleans mis-segmentation noise
-> edited 3DGS
```

### 关键模块

#### 1. Open-Vocabulary Object Grounding

作者把 prompt 拆成：

- object query
- editing action
- spatial relation
- color mapping

然后对同类候选对象先做几何/关系过滤，再做语言对齐。这比只看 embedding 相似度稳得多。

#### 2. 3D Gaussian Editing

五种操作都直接落在 Gaussians 上：

- 删除：直接移除目标 ROI 内高斯
- 换色：改颜色特征
- 添加 / 替换：由外部 Gaussian 生成模型产生新对象，再按 ROI 做几何拼接
- 移动：在世界坐标中轻量调整目标物体的高斯位置

#### 3. 编辑优化

由于 instance segmentation 在 ROI 边界附近可能出错，作者用 KNN 投票清理语义噪声，减少编辑污染到相邻对象。

### 关键实验信号

- 相较 `GaussCtrl`、`GaussianEditor`、`DGE`，3DSceneEditor 在 `TIS` 和 `IIS` 上都更高
- 初始编辑 `2-5 min`，后续针对同一 scene 的二次编辑低于 `1 min`，说明其 3D 语义缓存机制确实带来交互优势
- 用户研究中，对 `Prompt Alignment` 和 `Editing Quality` 的偏好都远高于对比方法
- Language-Object Correlation 模块在 `105` 组 object pairs 上，使用不同 VLM 组合仍有约 `88% - 94%` 的定位准确率

### 对当前 idea 的启发

这篇论文对你现在的方向非常有帮助，尤其是在 **where-to-edit** 这件事上：

- 它说明对象级编辑的关键先是 grounding，而不一定是更强的生成器
- 3D-only ROI grounding 非常适合作为后续 semantic localization 模块的工程参考
- 虽然它不是 feed-forward editor，但它给出了一个很实用的上限：如果你后面做前向编辑，至少在目标定位和 leakage 控制上要尽量接近这种 3D grounding 水平

### 实现约束

- 室内场景语义来自 `ScanNet200` 预训练 instance segmentation
- 室外场景使用 `OpenIns3D`
- prompt 解析使用 `GPT-4V`

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/WACV_2026/2026_3DSceneEditor_Controllable_3D_Scene_Editing_with_Gaussian_Splatting.pdf]]
