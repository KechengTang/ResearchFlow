---
title: "LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS"
venue: NeurIPS
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 3dgs
  - language-embedded-gaussians
  - open-vocabulary-grounding
  - compact-semantic-representation
  - sparse-coefficient-field
  - status/analyzed
core_operator: 把每个 Gaussian 的高维语言特征改写成全局字典上的稀疏系数字段，直接取消 LangSplat 的重型 decoder，并用 CUDA 优化的 sparse coefficient splatting 以超低维渲染代价恢复高维语言查询。
primary_logic: |
  输入多视图图像、SAM masks 与 CLIP 特征，先把每个 Gaussian 表示成全局字典上的稀疏系数而非低维 latent，再用 top-K 稀疏系数做高效 splatting，并通过矩阵乘法直接恢复高维语言特征，最终实现高帧率的开放词汇 3D 定位与分割。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2025/2025_LangSplatV2_High_dimensional_3D_Language_Gaussian_Splatting_with_450_FPS.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-20T19:23
updated: 2026-04-20T19:23
---

# LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv PDF](https://arxiv.org/pdf/2507.07136.pdf) / [Project Page](https://langsplat-v2.github.io/)
> - **Summary**: LangSplatV2 的贡献很聚焦: 它不再接受 LangSplat 的“先渲染低维 latent，再用大 decoder 回到 CLIP 空间”这一瓶颈，而是直接把高维语言特征写成稀疏系数场，从表示和 CUDA 渲染两侧同时把实时查询做通。
> - **Key Performance**:
>   - 在单张 A100 上，开放词汇 3D 查询从 LangSplat 的 `8.2 FPS` 提升到 `384.6 FPS`，高维特征 splatting 达到 `476.2 FPS`。
>   - 在 LERF 语义分割上达到 `59.9` overall IoU，高于 LangSplat 的 `51.4`；在 3D-OVS 上达到 `94.6` overall IoU。

## Part I / The "Skill" Signature

### 它到底在解决什么问题?
LangSplatV2 处理的是 LangSplat 这条路线最现实的部署问题:

**精度已经不错，但在线查询太慢，根本达不到真正实时。**

作者先做了 stage-wise time analysis，发现瓶颈非常集中:

- 渲染低维 latent 本身不慢。
- 真正拖垮速度的是把 latent 解码回高维 CLIP 特征的 heavyweight decoder。

所以这篇论文真正想要的能力边界是:

- **能做**: 高分辨率下实时开放词汇 3D 定位与分割。
- **不主打**: 重新发明 3D 语言场监督方式；它主要沿着 LangSplat 的语义建模继续提速。

### 输入 / 输出接口

- **输入**: 多视图图像、SAM masks、CLIP 特征、3DGS 场景。
- **输出**: 每个 Gaussian 的稀疏系数表示，以及高维语言特征渲染结果和查询结果。

### 边界条件

- 方法仍然继承 CLIP 语义表征的能力上限与偏置。
- 训练成本比 LangSplat 更高，它优化的是 test-time performance，不是训练轻量化。

## Part II / High-Dimensional Insight

### 方法真正新的地方

LangSplatV2 的判断很直接:

**既然 decoder 是速度瓶颈，那就不要再把 3D 语言场建模成“低维 latent + 在线解码”了。**

于是它改写了表示:

- 不再给每个 Gaussian 存一个待 decoder 还原的低维 latent。
- 而是把每个 Gaussian 看成全局语言字典上的一个 sparse code，只保存 top-K 稀疏系数。

同时它还改写了渲染:

- 渲染时只对 top-K 非零系数做 alpha blending。
- 最后通过矩阵乘法恢复高维语言特征，而不是过一个大 MLP decoder。

### The "Aha!" Moment

真正的 aha 在于:

**LangSplat 的瓶颈不是“3DGS 不够快”，而是“为了省显存而引入的低维 latent decoder 反而把在线查询拖死了”。**

LangSplatV2 于是反过来做:

- 不在 2D 上把 latent 解码回高维语义。
- 而是直接在 3D 里学高维语言特征的稀疏系数。

这条路线把问题从“高维特征太贵”改成“高维特征是否可以稀疏表示”。

因果链非常清楚:

`高维语义 -> 稀疏字典系数 -> top-K 稀疏渲染 -> 无重型 decoder -> 实时查询`

### 为什么这个设计有效?

- 稀疏字典假设允许模型保留高维语义表达力，同时避免直接 splat 512D 特征的高成本。
- 渲染阶段只处理 top-K 非零元素，时间复杂度从高维全量渲染变成稀疏渲染。
- 由于不再需要 decoder，在线推理链路被大幅缩短，且避免 decoder 误差。

### 策略权衡

- **优点**: 查询速度极快；分割与定位精度都比 LangSplat 更好或至少持平。
- **代价**: 训练更重；需要构建全局 codebook；方法仍然沿用 LangSplat 的静态语言场假设。

## Part III / Technical Deep Dive

### Pipeline

```text
multi-view images + SAM masks + CLIP features
-> build high-dimensional language supervision
-> represent each Gaussian as sparse coefficients over a global dictionary
-> keep top-K non-zero coefficients per Gaussian
-> CUDA sparse coefficient splatting
-> matrix multiplication restores high-dimensional language features
-> open-vocabulary 3D localization and segmentation
```

### 关键信号

- 论文先用时间剖析证明 decoder 才是主瓶颈，这让方法修改有很强的因果针对性。
- LangSplatV2 不是简单做工程优化，而是连表示形式都改了，所以才同时带来速度和精度收益。
- 与 LEGaussian 的比较也说明，它的优势不只是 codebook，而是 “3D 全局稀疏系数字段 + 无 decoder + sparse rendering” 这一整套组合。

### 少量关键数字

- LERF 上，LangSplat `122.1 ms / query, 8.2 FPS`；LangSplatV2 `2.6 ms / query, 384.6 FPS`。
- 高维 feature splatting 达到 `476.2 FPS`，相对 LangSplat 报告约 `42x` 渲染加速、`47x` 查询加速。
- LERF: 定位 `84.1` overall accuracy，分割 `59.9` overall IoU；LangSplat 分别为 `84.3` 与 `51.4`。
- 3D-OVS: `94.6` overall IoU，高于 LangSplat 的 `93.4`；Mip-NeRF360: `69.4`，高于 LangSplat 的 `57.3`。

### 实现约束

- 单张 A100 上，训练时间约 `3.0h`，高于 LangSplat 的 `1.0h`。
- 论文的重点明确偏向 test-time deployment，因此接受更高训练成本换实时查询能力。

## Local Reading / PDF 引用

![[paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2025/2025_LangSplatV2_High_dimensional_3D_Language_Gaussian_Splatting_with_450_FPS.pdf]]
