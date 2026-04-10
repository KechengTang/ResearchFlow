---
title: "MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting"
venue: ICCV
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - mesh-hybrid
  - dynamic-scene
  - simulation
  - mesh-adsorbed-gaussians
  - ARAP
  - SMPL
  - soft-body-simulation
  - status/analyzed
core_operator: 构造 mesh-adsorbed Gaussian 表示，让高斯在 mesh 附近“游走”而非刚性绑死；再用 MPE-Net、RMD-Net、RGD-Net 分别从 guide mesh 中提取姿态、细化 mesh 形变、预测高斯相对位移，从而统一重建与 simulation。
primary_logic: |
  输入单目视频并先预处理出时序一致的 coarse guide mesh，
  再通过 MPE-Net 从 guide mesh 中提取 mesh pose embedding，
  由 RMD-Net 预测 refined mesh deformation，RGD-Net 预测 Gaussian 在 mesh 上的相对位移，
  将两者组合为 final Gaussians 并进行 splatting-based rendering 监督，
  训练完成后可把 ARAP、SMPL、soft-body 等外部 mesh deformation priors 送入同一框架，实现重建与 simulation 统一。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_MaGS_Reconstructing_and_Simulating_Dynamic_3D_Objects_with_Mesh_adsorbed_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-10T20:11
updated: 2026-04-10T20:11
---

# MaGS: Reconstructing and Simulating Dynamic 3D Objects with Mesh-adsorbed Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://wcwac.github.io/MaGS-page/) | [ICCV PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Ma_MaGS_Reconstructing_and_Simulating_Dynamic_3D_Objects_with_Mesh-adsorbed_Gaussian_ICCV_2025_paper.pdf)
> - **Summary**: MaGS 不是简单把 mesh 和 Gaussian 拼在一起，而是让高斯在 mesh 附近保持“可吸附但可相对滑动”的状态，从而兼顾渲染灵活性和 simulation 所需的结构先验。
> - **Key Performance**:
>   - 在 D-NeRF 上平均达到 `PSNR 44.06 / MS-SSIM 0.9985 / LPIPS 0.0040`，超过现有强基线。
>   - 在 DG-Mesh dataset 上达到 `PSNR 40.76 / EMD 0.1106 / CD 0.6662`，仅用约 `981` faces，优化时间约 `47.6 min`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

MaGS 真正想回答的是：

**能不能用一个统一表示，同时服务于动态对象的高质量重建和后续 simulation，而不是先重建一个难以驱动的表示，再额外做昂贵转换？**

作者认为 reconstruction 和 simulation 的需求是天然冲突的：

- reconstruction 需要表达灵活、渲染强的表示
- simulation 需要结构清晰、可施加物理先验的表示

### 核心能力定义

- **输入**：单目视频，以及通过预处理得到的 guide mesh
- **输出**：高质量动态对象重建和可用于 simulation 的 mesh-Gaussian 混合表示
- **支持 simulation priors**：ARAP、SMPL、soft-body 等
- **不擅长**：拓扑变化很大的流体类 simulation

### 真正的挑战来源

- 把高斯 rigidly 绑在 mesh 上会牺牲渲染 fidelity
- 只靠高斯又难以承接有结构的 deformation prior
- simulation 时没有 timestamp，可不能继续依赖视频时序作为控制信号

### 边界条件

- 当前更偏 dynamic object，而不是一般 4D 大场景
- guide mesh 需要预处理生成，不是完全端到端前向方法

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

MaGS 的设计哲学很鲜明：

**让 mesh 负责“结构与先验”，让 Gaussians 负责“外观与渲染”，但两者不要刚性锁死，而是建立相对位移关系。**

这和 SuGaR / DG-Mesh 一类更强绑定的 mesh-Gaussian 方法不同。作者认为动态场景里，如果高斯被固定死在某个 facet 上，会导致 reconstruction fidelity 和 deformation rationality 二选一。

### The "Aha!" Moment

最关键的 aha 是：

**高斯不应该死贴在 mesh 表面，而应该允许在 mesh 局部坐标中以 barycentric + normal offset 的方式“游走”。**

这就是 `Mesh-adsorbed Gaussian` 的核心：

- 每个 Gaussian 记录所附着 facet 的 `MeshId`
- 用 barycentric coordinate `b` 和法向偏移 `h` 表示其相对 mesh 的位置
- mesh 变形时，Gaussian 可以跟随 mesh 运动，但仍保留局部自由度

### 为什么这个设计有效？

- `RMD-Net` 学 mesh deformation refinement，负责把粗 guide mesh 调成更符合视频的形变
- `RGD-Net` 学 Gaussian 相对 mesh 的偏移，负责在 mesh 约束下补足渲染 fidelity
- `MPE-Net` 不再依赖 timestamp，而是从 mesh pose 本身提 embedding，因此 simulation 阶段面对新形变也能泛化

### 战略权衡

- **优点**：真正打通 reconstruction 和 simulation；兼容多种 deformation prior
- **局限**：工程链路复杂，且像其他 mesh 方法一样不支持流体类的 changed-topology deformation

## Part III / Technical Deep Dive

### Pipeline

```text
monocular video
-> preprocess coarse temporally consistent guide mesh
-> MPE-Net extracts pose embedding from guide mesh
-> RMD-Net predicts relative mesh deformation
-> RGD-Net predicts relative Gaussian displacement on the deformed mesh
-> compose final Gaussians and render
-> jointly optimize mesh, Gaussians, and networks
-> at simulation time, replace guide mesh with ARAP / SMPL / soft-body deformed mesh
-> reuse learned priors for realistic rendering under novel deformations
```

### 关键模块

#### 1. Mesh-adsorbed Gaussian

每个 Gaussian 除了常规 3DGS 属性，还携带：

- `MeshId`
- `b = (b1, b2, b3)`：在 facet 上的重心坐标
- `h`：沿法线方向的偏移

这个表示既保留了结构关系，又赋予高斯足够自由度。

#### 2. RMD-Net / RGD-Net / MPE-Net

- `MPE-Net`：从 handle vertices 和 normals 中提取 mesh pose embedding
- `RMD-Net`：预测 relative mesh deformation
- `RGD-Net`：预测 Gaussian 在 mesh 变形下的相对位移

三者分工清楚，尤其是 MPE-Net 让系统摆脱了对时间戳的依赖。

#### 3. Mesh-guided Simulation

训练好之后，作者可以把用户编辑后的 mesh 或外部仿真器输出的 mesh 直接喂给系统，再由学到的 deformation priors 细化并渲染。这使其天然兼容：

- ARAP editing
- soft body collision
- SMPL editing

### 关键实验信号

- 在 D-NeRF 上，MaGS 平均 `PSNR 44.06`，高于 `SC-GS 43.31`
- 在 PeopleSnapshot 上，四个场景都优于 `SplattingAvatar`
- 在 DG-Mesh dataset 上，虽然 `CD` 不是最优，但 `EMD 0.1106` 和 `PSNR 40.76` 表现非常强，同时 mesh 复杂度远低于高面数方法
- ablation 中移除 `RMD-Net + RGD-Net` 会使 `PSNR` 从 `44.06` 掉到 `41.14`；移除 `Gaussian Hover` 也会明显退化

### 对当前 idea 的启发

MaGS 对你当前主线的启发，更多在未来扩展层面：

- 如果后面你想把 4D 编辑从 appearance/material 推到 geometry-aware 或 simulation-aware 的交互场景，这种 mesh-hybrid 方案很有潜力
- `timestamp-free deformation conditioning` 这点也很有意思，它说明 temporal generalization 不一定非得靠时间编码，也可以靠结构状态编码
- 但对你现在的第一版 feed-forward semantic editing 来说，它更适合作为“后续支线拓展”的参考，而不是第一批必须复用的底座

### 实现约束

- 所有实验运行在 `RTX 4090`
- mesh facet 数默认约 `1K`
- 依赖预处理生成 coarse guide mesh

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_MaGS_Reconstructing_and_Simulating_Dynamic_3D_Objects_with_Mesh_adsorbed_Gaussian_Splatting.pdf]]
