---
title: "DynaSurfGS: Dynamic Surface Reconstruction with Planar-based Gaussian Splatting"
venue: arXiv
year: 2024
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - dynamic-scene
  - dynamic-surface-reconstruction
  - planar-based-splatting
  - normal-regularization
  - ARAP
  - hexplane-deformation
  - mesh-extraction
  - status/analyzed
core_operator: 在 4DGS 的 HexPlane deformation backbone 上引入 planar-based Gaussian splatting，渲染 unbiased depth / normal 作为表面代理信号，再用 normal regularization 约束局部平面一致性、用 ARAP regularization 约束跨时间局部刚性，从而同时兼顾动态表面重建与渲染质量。
primary_logic: |
  输入带相机位姿与时间戳的动态多视图图像，
  先维护一组 canonical 3D Gaussians，并通过 HexPlane + MLP 预测每个时刻的 Gaussian deformation，
  再用 planar-based Gaussian splatting 渲染距离图、法线图与 unbiased depth，
  用深度恢复出的局部平面法线去约束 rasterized normal，得到更平滑且几何一致的动态表面，
  同时在不同时间随机采样两帧，对 kNN Gaussian 邻域施加 ARAP 约束以保持局部刚性，
  最终在保证 photorealistic rendering 的同时，配合 TSDF fusion 抽取出质量更高的动态 mesh。
pdf_ref: paperPDFs/4DGS_Reconstruction/arXiv_2024/2024_DynaSurfGS_Dynamic_Surface_Reconstruction_with_Planar_based_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-10T20:36
updated: 2026-04-10T20:36
---

# DynaSurfGS: Dynamic Surface Reconstruction with Planar-based Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://open3dvlab.github.io/DynaSurfGS/) | [arXiv 2408.13972](https://arxiv.org/abs/2408.13972)
> - **Summary**: DynaSurfGS 的关键不在于重新发明一套全新的动态表示，而是在 4DGS deformation pipeline 上补上一层“面感知”的几何监督: 用 planar-based Gaussian splatting 产出 unbiased depth 和 normal，再让局部平面法线与渲染法线对齐，同时用 ARAP 约束跨时间的局部刚性，从而把“高质量渲染”和“可用的动态表面重建”绑在一起。
> - **Key Performance**:
>   - 在 `DG-Mesh` dataset 上取得 `CD 0.910 / EMD 0.123 / PSNR 32.508`，几何指标接近甚至部分优于专门的 mesh 方法，同时渲染质量显著优于 `DG-Mesh`。
>   - 在 `D-NeRF` 上达到 `PSNR 34.31 / SSIM 0.9797 / LPIPS 0.0267`，相对 `4D-GS` 进一步提升 PSNR/SSIM，并显著改善表面光滑度。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

DynaSurfGS 盯住的是一个很具体但长期被 4DGS 系方法忽略的问题:
**能不能在保持动态 Gaussian 渲染质量的同时，真正恢复出平滑、几何可信、跨时间一致的动态表面？**

此前不少方法已经能把动态 novel-view rendering 做得很好，但它们更多是在优化“看起来像”，而不是优化“表面真的对”:

- `4D-GS` 一类方法更擅长渲染，抽出的 mesh 往往粗糙、噪声重。
- `DG-Mesh`、`MaGS` 这类 mesh-hybrid 方法更强调结构，但常常牺牲部分渲染表现，或者工程链条更重。
- 单纯把静态 surface reconstruction 正则搬到动态场景里，往往又缺少时间一致性约束。

### 核心能力定义

- **输入**: 带相机位姿与时间戳的动态多视图图像
- **输出**: 既能高质量渲染、又更利于抽取动态 mesh 的 4D Gaussian 表示
- **擅长**: 在动态场景里补强表面平滑性、几何一致性和局部时序刚性
- **不擅长**: 非刚性拓扑大变化、极少视角补全、真正的 feed-forward 实时重建

### 真正的挑战来源

- 3D Gaussians 本质是离散体元，天然不等价于连续表面。
- 动态场景里只做 photometric fitting 时，很容易得到好看但粗糙的 geometry。
- 如果只强推 surface smoothness，又会伤害渲染 fidelity，尤其在运动边界和细节区域。
- 跨时间表面若没有局部结构约束，很容易出现 temporal wobble 或邻域错位。

### 边界条件

- 方法仍然依赖 optimization-based dynamic 4DGS，而不是前向预测。
- 默认输入有可用的 camera poses 和 timestamps。
- mesh 是通过后处理抽取出来的，不是原生直接预测的显式 mesh。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

DynaSurfGS 的设计哲学很清晰:
**不要直接把“高质量动态 mesh 重建”当成一个完全独立的新表示问题，而是把它降解成“在现有 4DGS 渲染管线里注入更强的表面代理约束”。**

这意味着它保留了 4DGS/HexPlane 这条成熟的动态 deformation 主干，只在监督层面引入两类额外几何信号:

- 一类是基于 planar splatting 的 depth-normal consistency，用来让局部表面更像“面”而不是“点云团”。
- 一类是 ARAP 式的邻域刚性约束，用来让同一片局部结构在不同时间别散掉。

### The "Aha!" Moment

真正的 aha 在于:
**与其直接监督 mesh，不如先让 Gaussian renderer 自己输出更适合做 surface reasoning 的 unbiased depth / normal，再在图像域里做局部平面一致性约束。**

这一步非常关键，因为它把原本“离散高斯不好约束表面”的问题，转成了“深度与法线是否自洽”的问题:

- 渲染器先输出 normal map 与 distance map。
- 再从 unbiased depth 反推出局部平面法线。
- 最后最小化“深度恢复法线”和“高斯渲染法线”的差异。

这样模型不需要先有一个完美 mesh 才能学表面，而是在 3DGS 渲染过程中逐步长出更平滑、更 consistent 的 geometry。

### 为什么这个设计有效？

- **local planarity prior 很强**: 局部深度和法线一致，本质上就在鼓励表面平滑而不是散乱点云。
- **ARAP 补上了时间维约束**: 正常的 normal regularization 只管单帧 geometry，ARAP 才负责把邻域结构跨时间绑住。
- **与现有 dynamic 4DGS 兼容**: 不需要重写整个 4D representation，只是在 deformation + rendering 的基础上补 geometry-aware loss。
- **mesh extraction 受益直接**: 一旦 Gaussian 分布更贴近连续表面，后面的 TSDF / marching-style mesh 抽取自然更稳。

### 战略权衡

- **优点**: 非常适合作为动态表面先验层，能明显补强 geometry 而不完全放弃渲染质量。
- **局限**: 更像“优化阶段的几何校正器”，不是可扩展的 feed-forward 主干；并且 geometry 与 image quality 仍然存在张力。

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic multi-view images + camera poses + timestamps
-> initialize canonical 3D Gaussians
-> HexPlane + MLP predicts Gaussian deformation at each time
-> planar-based Gaussian splatting renders image / distance map / normal map / unbiased depth
-> local planar normals are recovered from unbiased depth
-> normal regularization aligns depth-derived normals with rendered normals
-> ARAP regularization constrains kNN Gaussian neighborhoods across time
-> optimize photometric + TV + normal + ARAP losses
-> extract dynamic mesh by TSDF fusion
```

### 关键模块

#### 1. HexPlane-based Dynamic Deformation

方法并没有丢掉 4DGS 常见的 deformation formulation，而是继续采用 `HexPlane + compact MLP` 来编码时空变化:

- HexPlane 负责提供 4D 时空特征
- MLP 解码 Gaussian 的位置、旋转、尺度偏移
- 每个时刻都由 canonical Gaussians 变形得到当前 frame 的 deformed Gaussians

这保证了它仍然是一条标准 dynamic GS 主线，而不是完全改换表示。

#### 2. Planar-based Gaussian Splatting

这一部分是方法的核心创新点。

作者借鉴 `PGSR` 的 planar-based splatting 思路，不只渲染 RGB，还渲染:

- 法线图 `N`
- 距离图 `L`
- 由二者推导出的 unbiased depth

然后基于局部平面假设，从深度图的邻域采样中恢复一个 depth-derived normal，再和渲染 normal 对齐。这个 normal regularization 直接把“表面应该平滑而且几何一致”的信号传回高斯参数。

#### 3. ARAP Regularization

为了防止动态场景里不同时间的局部结构飘散，作者对每个 Gaussian 建立 `k=10` 的邻域，并在随机采样的两个时间 `t1/t2` 之间施加 ARAP:

- 先通过 deformation field 得到两个时刻的 Gaussian centers
- 对每个点拟合一个局部刚性旋转
- 惩罚邻域相对位移在两个时刻的不一致

它的作用不是强行假设全局刚体，而是维持局部邻域的近似刚性。

#### 4. Mesh Extraction and Final Objective

训练目标由以下几项组成:

- `Lphoto`: RGB 重建
- `Ltv`: deformation field 的平滑约束
- `Lnormal`: 深度-法线几何一致性
- `Larap`: 时间上的局部刚性一致性

最终再用 `TSDF Fusion` 抽取 mesh，得到可视化和量化评估所需的几何结果。

### 关键实验信号

- 在 `DG-Mesh` dataset 上，`EMD 0.123 / PSNR 32.508` 说明它确实把 geometry 与 rendering 拉到了更平衡的位置。
- 在 `D-NeRF` 上，`PSNR 34.31 / SSIM 0.9797` 超过 `4D-GS` baseline，说明 geometry regularization 并没有把渲染彻底拖垮。
- ablation 很有价值:
  - 只有 `normal regularization` 时，geometry 更平滑，但 image fidelity 会掉。
  - 只有 `ARAP` 时，rendering 更强，但 geometry 还是粗。
  - 两者联合，才形成“几何平滑 + 时间一致 + 图像还不错”的折中。

### 对当前 idea 的启发

对你当前的 `4dgs semantic feedforward editing` idea 来说，DynaSurfGS 不是直接的骨干参考，但它提供了一个很有价值的“几何稳定器”视角:

- 如果后续要做 **语义局部编辑后的时空传播**，那么 surface-aware regularization 可以减少编辑区域在时间上的边界漂移。
- 如果前向模型要预测 editable 4DGS，`depth-normal consistency` 可以作为蒸馏目标或辅助 loss，让预测出来的高斯更贴近可编辑表面。
- 如果你准备把 semantic grounding 和 motion propagation 绑在一起，DynaSurfGS 提醒我们: 仅有语义定位还不够，局部几何支撑也会决定编辑结果是否稳定。

### 实现约束

- 这是 optimization-heavy 方法，不适合作为直接的实时 feed-forward 编辑骨干。
- 依赖 poses、timestamps 和较稳定的多视图输入。
- 表面正则会带来 geometry / appearance trade-off，参数调节敏感。
- mesh 抽取仍需要后处理，因此工程复杂度高于纯 4DGS 渲染方法。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/arXiv_2024/2024_DynaSurfGS_Dynamic_Surface_Reconstruction_with_Planar_based_Gaussian_Splatting.pdf]]
