---
title: "Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing"
venue: ECCV
year: 2024
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - appearance-editing
  - geometry-appearance-decoupling
  - texture-guided-control
  - real-time-rendering
  - status/analyzed
core_operator: 把 3DGS 的外观从高斯属性中剥离出来，改为“3D 几何 + 2D 纹理贴图”的显式解耦表示，让外观编辑落在纹理空间而不是逐高斯改颜色。
primary_logic: |
  输入一个已经优化好的 3DGS 场景，先用 UV mapping MLP 为高斯中心学习连续 UV 坐标，
  再用局部 Taylor 近似把 ray-Gaussian 交点高效映射到 2D 纹理域，
  最后用可学习纹理承载细粒度外观并在渲染时查询纹理，从而实现几何保持下的外观编辑。
pdf_ref: paperPDFs/3DGS_Editing/ECCV_2024/2024_Texture_GS_Disentangling_the_Geometry_and_Texture_for_3D_Gaussian_Splatting_Editing.pdf
category: 3DGS_Editing
created: 2026-04-18T14:30
updated: 2026-04-18T14:30
---

# Texture-GS: Disentangling the Geometry and Texture for 3D Gaussian Splatting Editing

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2403.10050) / [PDF](https://arxiv.org/pdf/2403.10050.pdf)
> - **Summary**: Texture-GS 的核心价值不是再做一种“更会改图”的 3DGS，而是把 3DGS 里原本纠缠在一起的几何与外观拆开，让贴图替换、局部换肤、外观迁移这类编辑第一次真正有了清晰的作用位点。
> - **Key Performance**:
>   - 论文在 DTU 上展示了全局纹理替换和局部外观编辑，同时保持几何结构基本不被破坏。
>   - 渲染速度约为 58 FPS，说明这种解耦并没有把 3DGS 的实时性优势牺牲掉。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

Texture-GS 针对的是 3DGS 编辑里一个很典型但经常被忽略的结构性瓶颈：
**在原始 3DGS 里，外观和几何都编码在高斯属性里，所以一旦你想改纹理，往往会顺手把几何一致性也一起扰乱。**

这会直接限制几类重要编辑：

- 纯外观替换，例如 texture swapping
- 局部材质修改而不改变形状
- 几何应保持稳定的风格或颜色编辑

它的输入输出接口很直接：

- 输入：已有的 3DGS 场景
- 中间表示：显式几何 + 可学习 2D 纹理
- 输出：在几何基本保持下可实时渲染的编辑结果

边界也很清楚：

- 它主要解决的是 **appearance editing**
- 对大形变、几何重建、拓扑修改帮助有限

## Part II / High-Dimensional Insight

### 方法真正新在哪里

Texture-GS 的关键不是“往 3DGS 上再挂一个贴图”，而是把编辑接口从高斯属性空间移到了 **纹理空间**。

一旦这个迁移成立，很多编辑都会变得自然：

- 要改外观，就改纹理
- 要保几何，就尽量别直接改高斯几何参数
- 要维持效率，就让纹理查询和光栅化仍然贴着 3DGS 的实时渲染路径走

### The "Aha!" Moment

真正的 aha 在于：

**3DGS 里最难编辑的不是“外观不够可控”，而是“外观没有被单独表示出来”。**

Texture-GS 的回答是：

1. 用 UV mapping MLP 给高斯中心建立 2D 纹理坐标
2. 用局部 Taylor 近似高效计算 ray-Gaussian 交点对应的 UV
3. 用 learnable texture 存细粒度外观

这样外观编辑就从“改一堆 3D 高斯颜色参数”变成了“改一张连续纹理图”。

这对你的研究尤其有价值，因为它提供了一个很清楚的设计模式：
**如果编辑目标本质上是外观，不要在 3D 原语参数里硬改，应该先把外观作用域剥离出来。**

### 权衡与局限

- 优势：编辑作用位点清晰，几何保持更好，渲染仍快
- 局限：依赖稳定的几何与映射学习，对几何编辑本身帮助不大

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS scene
-> UV mapping MLP for Gaussian centers
-> local Taylor approximation for ray-Gaussian intersections
-> learnable 2D texture stores appearance
-> render with decoupled geometry and texture
-> appearance-edited 3DGS scene
```

### 关键技术点

#### 1. Geometry / Texture Decoupling

论文把几何和外观拆成两个层次：

- 几何仍由 3DGS 负责
- 外观交给 2D 纹理域

这让外观操作不再直接污染几何参数。

#### 2. Efficient UV Query

直接为每个交点跑一次复杂映射会很慢，所以作者用局部 Taylor 近似加速交点 UV 计算，保住了实时渲染。

#### 3. Cycle Consistency

论文还引入了 3D 空间的 cycle consistency 约束，避免 UV 映射退化成不稳定或不可逆的局部编码。

### 关键信号

- 在 DTU 上，论文展示了它不仅能做新视角合成，也能做全局纹理替换与局部外观编辑。
- 58 FPS 这个信号很重要，因为它说明“显式外观解耦”没有把 3DGS 的核心工程优势丢掉。

### 对你的价值

如果你后面做 4DGS 编辑，这篇论文最值得迁移的不是某个具体损失，而是：

- **先分清楚编辑是在改几何还是改外观**
- **外观编辑应该尽量落在独立表示上**

这对动态场景尤其关键，因为 4D 情况下如果几何和外观还纠缠在一起，时间一致性问题会被进一步放大。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ECCV_2024/2024_Texture_GS_Disentangling_the_Geometry_and_Texture_for_3D_Gaussian_Splatting_Editing.pdf]]
