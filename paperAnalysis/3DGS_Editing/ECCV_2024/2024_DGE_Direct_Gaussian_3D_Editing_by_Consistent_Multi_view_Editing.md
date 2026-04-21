---
title: "DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing"
venue: ECCV
year: 2024
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - scene-editing
  - direct-3d-fitting
  - multi-view-consistency
  - epipolar-constraint
  - spatio-temporal-attention
  - partial-editing
  - status/analyzed
core_operator: 先把多视角渲染序列当作 orbital video 做一致图像编辑，再用带 epipolar 约束的特征传播生成可直接监督 3DGS 的一致视图，最后一次性直接拟合高斯场。
primary_logic: |
  输入源 3DGS 的多视角渲染与文本编辑指令，
  先对关键视角使用带 spatio-temporal attention 的多帧编辑，
  再用 epipolar 约束的跨视角特征注入把编辑传播到其余视角，得到多视角一致的编辑图像，
  最后直接用 LPIPS 等图像损失拟合 3DGS，而不是反复做编辑-重建循环。
pdf_ref: paperPDFs/3DGS_Editing/ECCV_2024/2024_DGE_Direct_Gaussian_3D_Editing_by_Consistent_Multi_view_Editing.pdf
category: 3DGS_Editing
created: 2026-04-18T16:15
updated: 2026-04-18T16:15
---

# DGE: Direct Gaussian 3D Editing by Consistent Multi-view Editing

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ECCV 2024 PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09386.pdf) · [Project Page](https://silent-chen.github.io/DGE/)
> - **Summary**: DGE 的核心主张是，不要再让 3D 编辑靠“反复编辑单张图再慢慢平均进 3D”去收敛，而是先在图像空间直接生成一组多视角一致的编辑结果，再一次性直接拟合 3DGS，因此同时提升速度、保真度和局部可控性。
> - **Key Performance**:
>   - `CLIP similarity = 0.226`、`CLIP directional similarity = 0.67`、平均编辑时间约 `4 min`，优于 `GaussianEditor` 的 `0.201 / 0.60 / 7 min`。
>   - 只做 `400 iterations` 时已能在 `2 min` 内得到可用结果，显示直接拟合路线明显更快收敛。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文瞄准的是文本驱动 3D 编辑中的一个效率瓶颈：

**现有方法通常都在“单视图编辑”和“3D 重建更新”之间来回迭代，而这个回路慢的根源并不是 3D 优化本身，而是输入给 3D 的多视角编辑结果彼此并不一致。**

无论是 iterative dataset update 还是 SDS，本质上都在尝试从冲突的 2D 信号中慢慢逼近一个 3D 共识，所以编辑一条 prompt 常常要数分钟甚至更久。

### 核心能力定义

- **输入**：源 3DGS、文本编辑指令，以及可选的局部编辑区域。
- **输出**：与目标文本一致的编辑后 3DGS。
- **额外能力**：支持 selective editing，也就是只改局部 Gaussians。
- **主要目标**：高保真、高效率、可局部编辑。

### 真正的挑战来源

- 单目 2D 编辑器对不同视角 independently sample，几乎不可能天然生成相互匹配的结果。
- 3D 优化如果直接吃这些不一致图像，就只能在“平均矛盾”中慢慢收敛，结果容易糊。
- 若想做局部编辑，还得能找到哪些 Gaussians 应该被允许更新。

### 边界条件

- 依赖已有相机位姿、标定和一个可稳定渲染的源 3DGS。
- 局部编辑仍然依赖 2D mask 再回投到 3D 的流程。
- 方法虽然比 prior work 快很多，但仍然是优化式，不是完全 feed-forward。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

DGE 的思路非常直接：

**先把“多视角一致编辑”这件事在图像空间做对，再让 3DGS 去拟合正确的监督，而不是让 3DGS 在错误监督里替你擦屁股。**

它因此把整个问题拆成两段：

- 多视角一致图像编辑。
- 对这组一致图像做 direct 3D fitting。

### The "Aha!" Moment

真正的 aha 是：

**把一组围绕静态物体的多视角图像看成一段 orbital video。**

这样一来，3D 多视角一致性就能转写成视频编辑里的时序一致性问题。作者据此做了两件关键事：

- 对关键视角联合做 `spatio-temporal attention`，让编辑器不再逐帧独立工作。
- 把关键视角的编辑特征传播到其他视角时，不是全平面搜对应，而是用 `epipolar constraint` 把匹配限制在极线上，减少错误传播。

于是 DGE 不是靠多轮迭代把不一致“磨平”，而是从一开始就尽量生成能直接喂给 3DGS 的一致监督。

### 为什么这个设计有效？

- 把多视角当视频后，编辑器终于能共享跨帧信息，而不再是每张图随机发挥。
- `epipolar constraint` 利用 3D 已知相机几何，把 correspondence 搜索从整个 2D 平面缩到一条线，尤其在纹理相似、外观模糊时更可靠。
- 一旦图像监督本身足够一致，3DGS 就可以直接 fit 到这些结果上，而不需要很多轮 edit-retrain 循环。
- 3DGS 的显式性还天然支持 selective editing，只更新指定区域即可。

### 战略权衡

- **优点**：速度和 fidelity 同时提升，而且比 NeRF 方案更适合做局部编辑。
- **代价**：方法仍要做多视角渲染、扩散编辑和 3D 优化，且依赖相机几何质量。
- **能力边界**：极细纹理仍可能需要少量 iterative refinement，作者也保留了再次渲染再编辑的可选循环。

## Part III / Technical Deep Dive

### Pipeline

```text
source 3DGS
-> render multi-view images
-> reorder views as an orbital video
-> jointly edit key views with spatio-temporal attention
-> inject edited key-view features to other views with epipolar-constrained correspondences
-> obtain roughly consistent edited multi-view images
-> directly fit 3DGS with LPIPS / image losses
-> optional refinement iteration
-> optional local editing by masking target Gaussians
```

### 关键模块

#### 1. Multi-view consistent editing

作者只对随机采样出的关键视角启用多帧联合 attention，控制计算量。非关键视角则通过 feature injection 接收关键视角的编辑信息，因此不用对所有视图都跑昂贵的全局联合 attention。

#### 2. Epipolar-constrained feature injection

这里的关键不是“做特征传播”，而是 **传播时显式用 3D 几何约束 correspondence**。这样传播出的编辑信号更可能落到同一个 3D 区域，而不是被纹理相似但几何错误的位置吸走。

#### 3. Direct Gaussian fitting

拿到一致编辑图像后，DGE 直接用图像损失拟合 3DGS。作者特别强调 LPIPS 在这里有价值，因为它对小幅跨视图差异更鲁棒，不会像逐像素硬对齐那样放大小噪声。

#### 4. Partial editing

局部编辑通过 2D segmentation 再回投到 3D Gaussian mask 来做，只允许被选中的 Gaussians 更新。这让 DGE 在场景中能只改目标物体，不必全局受影响。

### 关键实验信号

- 与 `Instruct-N2N`、`ViCA-NeRF`、`GaussianEditor`、`IP2P + SDS` 相比，DGE 在两项 CLIP 指标上都最好，同时编辑时间最短。
- 消融显示，没有 multi-view consistency 时，CLIP 与 directional score 都明显下降，说明 direct fitting 的前提确实是先把图像监督做一致。
- 再加上 epipolar constraint 后，整体分数继续小幅提升，更重要的是细节纹理和 correspondence 更稳定。
- 与 `GaussianEditor` 的迭代对比表明，DGE 在很少迭代下就能出结果，而前者收敛更慢。

### 少量关键数字

- `CLIP similarity`: `0.226`
- `CLIP directional similarity`: `0.67`
- 平均编辑时间：约 `4 min`
- `400 iterations` 无 refinement 时：`under 2 min`

### 实现约束

- 评测使用 `3` 个场景、`10` 条 prompts，并从训练视角中随机采样 `20` 个相机位姿做 CLIP 评估。
- 3DGS 优化通常做 `500-1500 iterations`。
- 作者在 3D 拟合阶段使用 `LPIPS + L1` 风格的图像监督。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ECCV_2024/2024_DGE_Direct_Gaussian_3D_Editing_by_Consistent_Multi_view_Editing.pdf]]
