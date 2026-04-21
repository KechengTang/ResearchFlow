---
title: "Gaussian Grouping: Segment and Edit Anything in 3D Scenes"
venue: ECCV
year: 2024
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - open-world-segmentation
  - instance-segmentation
  - identity-encoding
  - local-gaussian-editing
  - object-removal
  - object-inpainting
  - status/analyzed
core_operator: 给每个 Gaussian 增加可学习的 Identity Encoding，并用 SAM 掩码监督加 3D 邻域正则把 2D anything masks 提升成可编辑的 3D grouping 表示。
primary_logic: |
  输入多视角图像与 SAM 自动生成的 2D masks，
  先跨视角关联 2D mask 的实例 ID，
  再把每个 Gaussian 的 Identity Encoding 通过可微 splatting 渲染到 2D，并用 2D identity 分类损失与 3D 邻域一致性正则共同训练，
  最后把分组后的高斯场直接当作可组合的编辑单元，实现删除、重排、上色、风格迁移和补洞等局部操作。
pdf_ref: paperPDFs/3DGS_Editing/ECCV_2024/2024_Gaussian_Grouping_Segment_and_Edit_Anything_in_3D_Scenes.pdf
category: 3DGS_Editing
created: 2026-04-18T16:55
updated: 2026-04-18T16:55
---

# Gaussian Grouping: Segment and Edit Anything in 3D Scenes

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ECCV 2024 PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04195.pdf) · [GitHub](https://github.com/lkeab/gaussian-grouping)
> - **Summary**: Gaussian Grouping 的核心不是直接做文本编辑，而是先把 3DGS 从“只会重建外观和几何”推进到“还能把场景拆成可操作的对象组”，这样后续 removal / inpainting / recomposition / style transfer 都能直接在高斯组上做，而不必每次从全局 3D 优化重新开始。
> - **Key Performance**:
>   - 在 LERF-Mask 的 open-vocabulary segmentation 上达到 `69.7 / 77.0 / 71.7 mIoU`，明显高于 `LangSplat 52.8 / 50.4 / 69.5` 等基线。
>   - 3D object inpainting 对比 `SPIn-NeRF` 的 `5h` 训练，作者报告自身方案约 `1h train + 20min finetune` 即可得到更好的补洞质量。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文真正想解决的是：

**3DGS 虽然重建快、渲染快，但它不知道“哪个高斯属于哪个物体或哪块 stuff”，因此几乎不具备真正的对象级场景编辑能力。**

所以 Gaussian Grouping 的目标不是单纯提升编辑质量，而是先补上一个更基础的能力层：

**把 open-world 2D segmentation 提升成可跨视角一致、可直接操作的 3D grouping。**

### 核心能力定义

- **输入**：多视角图像以及 SAM 自动生成的 2D anything masks。
- **输出**：同时具备重建与 instance / stuff grouping 能力的 3DGS。
- **直接带来的能力**：对象删除、对象位置交换、颜色修改、风格迁移、基于删除后的补洞。
- **特别擅长**：把“分割”直接变成“编辑接口”，使不同对象组能被独立操作。

### 真正的挑战来源

- 2D SAM masks 本身是逐视角产生的，不自带跨视角一致 ID。
- NeRF 一类隐式表示即便学到语义，也不容易把“这个对象”映射成一组可直接操控的参数。
- 3D 场景中有遮挡、重叠和细碎结构，仅靠 2D identity supervision 很难稳定监督那些很少直接可见的高斯。

### 边界条件

- 该方法主要提供的是 **可编辑的 3D grouping 表示**，文本驱动生成编辑并不是它的主目标。
- 依赖 SAM 的 2D mask 质量和跨视角 mask ID 关联质量。
- 动态场景或时间变化对象不在其核心能力范围内。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

Gaussian Grouping 的设计哲学很鲜明：

**不要试图在 3DGS 外面再套一个语义网络做后处理，而是直接把“对象身份”当成 Gaussian 的一等属性一起学。**

作者没有改高斯的颜色、位置或几何建模方式，而是额外给每个 Gaussian 增加一个 `Identity Encoding`，把“它是谁”变成和“它长什么样”同层级的参数。

### The "Aha!" Moment

真正的 aha 是：

**把 3D 场景分组问题改写成“高斯级 identity feature 渲染问题”。**

具体做法是：

- 每个 Gaussian 携带一个低维 Identity Encoding。
- 像渲染颜色一样，把这些 identity feature 通过可微 splatting 渲染回 2D。
- 在 2D 上做 identity classification，反向更新 3D 高斯的 identity。

这样，2D SAM supervision 就不再只是给 2D 图像打标签，而是被真正 lift 到 3D 高斯层。

### 为什么这个设计有效？

- 3DGS 本身就是显式离散表示，每个高斯天然适合作为“对象组成单元”。
- Identity Encoding 不需要很高维，文中用 `16` 维就能支撑细粒度 grouping，训练和渲染都比较轻。
- 仅有 2D supervision 时，那些被遮挡或很少可见的高斯容易学不好；因此作者再加 `3D Regularization Loss`，迫使 3D 邻域内的高斯 identity 接近，从而把可见区域的监督扩展到隐蔽区域。
- 一旦对象组被稳定学出，编辑就能退化成简单的组级操作，而不是再次全局优化。

### 战略权衡

- **优点**：把分割与编辑统一进同一套高斯表示，后续局部操作非常直接。
- **代价**：它更多是“编辑基础设施”，并不直接解决文本生成、多视角扩散一致性等更高层问题。
- **定位**：这是 3DGS 编辑系统里的“对象可寻址层”，特别适合作为后续局部编辑框架的底座。

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view images
-> SAM anything masks per view
-> cross-view mask ID association
-> add Identity Encoding to each Gaussian
-> render identity features back to 2D via differentiable splatting
-> optimize with 2D identity loss + 3D regularization loss
-> grouped 3D Gaussians
-> local Gaussian editing by group deletion / relocation / color SH tuning / new Gaussian insertion
```

### 关键模块

#### 1. Identity Encoding

每个 Gaussian 新增一个长度为 `16` 的 Identity Encoding。和 view-dependent 颜色不同，identity 不该随视角变化，因此作者把它的 SH 阶数设为 `0`，只保留 direct-current 成分。这个设计很关键，因为它明确把“身份一致性”从“外观变化”里剥离了出来。

#### 2. 2D Identity Loss

渲染后的 identity feature 会过一个线性层映射回全部 mask IDs，再做交叉熵分类。这个损失不是直接监督 3D 高斯，而是借助 2D differentiable rendering 间接监督 3D grouping。

#### 3. 3D Regularization Loss

作者显式用 3D 邻域一致性去弥补 2D mask supervision 的稀疏和不完整。对每个采样高斯，要求其与 3D 空间里 top-k 邻居的 identity distribution 更接近，从而让对象内部和被遮挡部分也能得到稳定 grouping。

#### 4. Local Gaussian Editing

这是这篇论文最实用的地方。分组完成后：

- `removal`：直接删该组 Gaussians
- `recomposition`：交换组中心位置
- `colorization`：只调该组 SH 颜色
- `style transfer`：进一步放开位置和尺寸
- `inpainting`：先删组，再只对新增少量 Gaussians 做 LaMa 引导微调

它把编辑操作真正落实成了 group-wise Gaussian operation list。

### 关键实验信号

- 在 open-vocabulary segmentation 与 panoptic novel-view segmentation 上都优于 NeRF 系和近期 Gaussian 方法，说明它不只是能重建，还确实学到了更好的 3D grouping。
- 编辑实验里，removal / inpainting / style transfer / multi-object concurrent edits 都能直接依赖分组后的表示完成。
- 作者特别强调，很多操作如 removal、recomposition、colorization 本身几乎不需要再训练，只是直接操作高斯组。

### 少量关键数字

- LERF-Mask：`69.7 / 77.0 / 71.7 mIoU`
- Replica / ScanNet novel-view panoptic：`71.15 / 68.70 mIoU`
- object inpainting：约 `1h train + 20min finetune`

### 实现约束

- 2D mask ID 关联相较 cost-based linear assignment 提速超过 `60x`。
- Identity Encoding 维度最终取 `16`，作者认为在精度与效率之间更平衡。
- 3D inpainting 仅对新增少量 Gaussians 进行监督，而不是重新训练整个场景。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ECCV_2024/2024_Gaussian_Grouping_Segment_and_Edit_Anything_in_3D_Scenes.pdf]]
