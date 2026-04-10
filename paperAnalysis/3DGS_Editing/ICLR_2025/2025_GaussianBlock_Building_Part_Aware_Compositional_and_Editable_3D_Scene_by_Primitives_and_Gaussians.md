---
title: "GaussianBlock: Building Part-Aware Compositional and Editable 3D Scene by Primitives and Gaussians"
venue: ICLR
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - compositional-reconstruction
  - part-aware
  - editable-scene
  - superquadrics
  - hybrid-representation
  - semantic-disentanglement
  - building-block-editing
  - status/analyzed
core_operator: 用 superquadrics primitives 做语义可编辑的 coarse building blocks，再绑定 3D Gaussians 做细节皮肤，并通过 attention-guided centering、dynamic split/fuse 和 binding inheritance 维持语义一致的 part-level disentanglement。
primary_logic: |
  输入多视图图像与 silhouette，
  先优化语义引导的 superquadrics primitives，借助双重可微渲染得到 2D prompts，再通过 attention-guided centering loss、动态 splitting 与 fusion 得到语义一致的 part-aware blocks，
  随后将 3D Gaussians 绑定到 primitive 三角面上并在局部空间优化，
  通过 identity inheritance 与位置正则保持两者连接，
  最终获得既可精细渲染又能像积木一样进行物理式编辑的混合表示。
pdf_ref: paperPDFs/3DGS_Editing/ICLR_2025/2025_GaussianBlock_Building_Part_Aware_Compositional_and_Editable_3D_Scene_by_Primitives_and_Gaussians.pdf
category: 3DGS_Editing
created: 2026-04-10T19:43
updated: 2026-04-10T19:43
---

# GaussianBlock: Building Part-Aware Compositional and Editable 3D Scene by Primitives and Gaussians

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://qpkl9034.github.io/) | [arXiv 2410.01535](https://arxiv.org/abs/2410.01535)
> - **Summary**: GaussianBlock 的核心不是继续把所有能力压进一个 entangled 的 Gaussian field，而是把可编辑性前置到表示层: 用 superquadrics 做语义一致的 building blocks，再用高斯做细节修补，形成可拆、可拖、可局部改的 hybrid 3D representation。
> - **Key Performance**:
>   - 在 DTU primitives reconstruction 上平均 `CD 3.69`，优于 `DBW 4.78` 和 `EMS 5.67`。
>   - 在混合表示上实现 part-aware decomposition 与 editable scene 的同时，仍保持具有竞争力的重建质量，如 DTU 上表格给出 `SSIM 94.3 / PSNR 30.7 / LPIPS 10.3`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

GaussianBlock 关注的是一个和许多编辑方法不同的问题：

**能不能从一开始就把 3D 场景重建成“语义可拆分、结构可操作”的表示，而不是先重建一个高保真但缠在一起、事后再艰难编辑的表示？**

作者认为传统 NeRF/GS 的问题不只是不好编辑，而是底层表示本身就高度 entangled：

- 语义边界不清楚
- 局部结构不可解释
- 很难像真正的零件那样进行物理式操作

### 核心能力定义

- **输入**：多视图图像及其 silhouette
- **输出**：part-aware superquadrics + bound Gaussians 的混合场景表示
- **支持能力**：像搭积木一样做部分级删除、旋转、平移等可控编辑
- **不主打**：纯粹追求极致渲染 fidelity 或复杂背景建模

### 真正的挑战来源

- 纯 primitives 可编辑但保真差
- 纯 3D Gaussians 保真高但表示纠缠、难以局部精确改动
- 若没有语义约束，primitive decomposition 往往只是几何拟合，不代表真实可编辑部件

### 边界条件

- 当前更适合 object-centric 或由 silhouette 主导的场景
- 背景建模仍较弱，作者也明确把这点列为限制

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

GaussianBlock 的设计非常有意思：

**用 primitives 负责“可操作语义结构”，用 Gaussians 负责“高保真外观细节”。**

这等于承认两类表示各有短板：

- superquadrics 非常适合做积木式编辑
- Gaussians 非常适合做高质量重建

与其二选一，不如明确分工，然后把它们绑在一起。

### The "Aha!" Moment

最核心的 aha 是：

**真正可编辑的 3D 表示，不应该只依赖事后语义分组，而要在重建阶段就通过语义先验把 primitive 学成“有意义的部件”。**

为此作者引入 `Attention-guided Centering (AC) loss)`：

- 先从 superquadrics 通过 dual rasterization 生成可微 point / box prompts
- 再借助 2D 预训练分割模型得到 attention maps
- 对同语义查询的 attention cluster 做聚心约束

结果就是 primitive 不再只是几何上勉强贴合，而会更倾向于形成语义一致的 part blocks。

### 为什么这个设计有效？

- split / fuse 机制避免 primitive 卡在坏局部最优：过大就拆、语义重叠就并
- Gaussians 绑定到 primitive 三角面上后，可以在保持 part identity 的同时补回细节
- identity inheritance 让高斯在 densification 后仍记得“自己属于哪个 primitive、哪个 triangle”

### 战略权衡

- **优点**：表示层就具备 editability，特别适合做部分级操控
- **局限**： fidelity 仍很难完全追平纯高斯重建；同时对复杂背景与开放场景支持有限

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view images + silhouettes
-> optimize superquadrics primitives
-> dual rasterization derives differentiable point/box prompts
-> 2D segmentation attention supervises attention-guided centering
-> dynamic splitting and fusion improve granularity and compactness
-> initialize Gaussians on primitive triangles
-> localized Gaussian optimization with binding inheritance and position regularization
-> editable hybrid scene representation
```

### 关键模块

#### 1. Attention-Guided Centering Loss

这部分是整篇的语义来源。作者不是直接用 2D mask 投 3D label，而是把 superquadrics 渲染成 differentiable prompts 去查询 2D segmentation model，再根据 attention clusters 把 primitive 往语义中心拉。

#### 2. Dynamic Splitting and Fusion

如果一个 primitive 同时覆盖多个语义中心，就拆开；如果多个 primitive 语义相同或高度重叠，就融合。这样能同时提升：

- semantic coherence
- compactness
- 后续编辑时的可解释性

#### 3. Gaussian Binding and Inheritance

高斯初始化在 primitive 三角面中心，并额外保存：

- 绑定到哪个 primitive
- 绑定到哪个 triangle

之后就可以在 primitive 发生旋转/平移/删除时，让相关高斯跟着一起动，从而实现真正的 part-level physical editing。

### 关键实验信号

- DTU 上 primitives reconstruction 的 `CD` 明显优于 `DBW`
- part-aware decomposition 可视化显示其语义拆分比 `DBW` 更合理、比 `OmniSeg3D` 更细粒度
- 在 fidelity 对比中，GaussianBlock 虽不及纯 `3DGS`，但远好于 primitive-only 的 `DBW`，说明 hybrid design 确实把 editable 表示和保真度做了折中
- ablation 中移除 `LAC` 会让 `CD` 从 `3.69` 恶化到 `4.75`；移除 splitting 或 fusion 也会显著影响质量或 compactness

### 对当前 idea 的启发

这篇论文对你当前方向的价值，在于它从表示层回答了一个很重要的问题：

- 如果想做局部、精确、可解释的编辑，**表示最好天然支持 part-level disentanglement**
- 语义一致的 coarse blocks + 高保真 fine skin，这种两层结构也许可以迁移到未来的 canonical edit 表示中
- 它不是 feed-forward editor，但它很好地说明了“where-to-edit”如果直接由表示层提供，会比纯后验语义分割更稳

### 实现约束

- 两阶段优化：先 primitive，再 Gaussian binding
- 关键超参包括 `γ = 0.2`、`β = 0.7`、`ϵpos = 0.5`
- 当前主要处理 object-centric 场景，背景能力有限

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICLR_2025/2025_GaussianBlock_Building_Part_Aware_Compositional_and_Editable_3D_Scene_by_Primitives_and_Gaussians.pdf]]
