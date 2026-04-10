---
title: "AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting"
venue: arXiv
year: 2026
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - feed-forward
  - multimodal-stylization
  - appearance-editing
  - text-guided-editing
  - image-guided-editing
  - view-consistency
  - status/analyzed
core_operator: 冻结预训练 3D 重建骨干，只在独立 style branch 中用 Long-CLIP 条件与 zero-conv 残差注入风格，再与原始几何参数重组，单次前向生成风格化 3DGS。
primary_logic: |
  输入无位姿多视图内容图像与文本/风格图风格条件，
  先用冻结的 AnySplat backbone 恢复相机与几何，
  再在复制出来的 Aggregator 与 Gaussian Head 中注入 Long-CLIP 风格嵌入，只学习外观相关分支，
  最后由 Gaussian Adapter 将原始几何与风格化颜色/旋转组合，得到可直接渲染的 stylized 3DGS。
pdf_ref: paperPDFs/3DGS_Editing/arXiv_2026/2026_AnyStyle_Single_Pass_Multimodal_Stylization_for_3D_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-10T18:24
updated: 2026-04-10T18:24
---

# AnyStyle: Single-Pass Multimodal Stylization for 3D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [GitHub](https://github.com/joaxkal/AnyStyle) | [arXiv 2602.04043](https://arxiv.org/abs/2602.04043)
> - **Summary**: AnyStyle 把 feed-forward 3DGS 风格化从“只能吃一张风格图”的图像迁移，推进到“文本/图像统一条件、冻结几何骨干、单次前向完成”的可组合外观编辑框架。
> - **Key Performance**:
>   - 在 `Train / Truck / M60 / Garden` 四个场景上，`AnyStyle_img` 的 `ArtFID` 分别为 `22.86 / 22.95 / 22.81 / 22.32`，是表中最强的 feed-forward 风格匹配结果。
>   - `AnyStyle_txt` 在四个场景上的 `ArtScore` 为 `9.87 / 10.56 / 10.20 / 10.46`；推理速度约 `0.07s`，teaser 中给出单张内容图风格化低于 `0.1s`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

AnyStyle 想解决的不是“怎么把 3D 场景做成艺术风格”这么宽泛的问题，而是更具体的一点：

**能不能在不重新训练整个 3D 重建系统的前提下，把任意无位姿多视图场景做成可控、零样本、单次前向的 3DGS 风格化？**

此前的 feed-forward 3D 风格化方法虽然已经把速度做上去了，但仍有两个明显瓶颈：

- 风格控制几乎都依赖单张 style image，语义可控性弱
- 风格模块与几何骨干往往强耦合，迁移到新骨干时要重训整套模型

### 核心能力定义

- **输入**：无位姿多视图内容图像，以及文本提示或风格参考图
- **输出**：可直接渲染的 stylized 3D Gaussian scene
- **擅长**：全局外观、材质、笔触、色彩氛围层面的快速风格迁移，以及文本-图像间的连续插值
- **不擅长**：几何结构编辑、局部对象级精确修改、动态 4D 场景建模

### 真正的挑战来源

- 几何重建与风格迁移往往会互相干扰，尤其是外观分支修改过深时会破坏结构
- 文本风格与图像风格的控制接口不同，直接并列训练容易出现模态缝隙
- 多视图风格化若只在 2D 层处理，很容易丢掉 3D 一致性

### 边界条件

- 方法建立在预训练 AnySplat 之上，本质上是 **appearance stylization adapter**，不是新的 3D 重建主干
- 论文关注的是静态 3DGS 的风格控制，不处理时序一致性问题

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

AnyStyle 的设计核心非常清楚：

**把“几何恢复”和“风格注入”彻底拆开。**

也就是说，作者不再让一个统一网络同时承担：

- 相机与几何恢复
- 风格条件理解
- 外观生成

而是先用冻结的 backbone 保住结构，再让独立 style branch 只负责把风格写进高斯外观参数。

### The "Aha!" Moment

真正的关键直觉是：

**如果预训练重建骨干已经能稳定恢复 3D 结构，那么风格控制最好表现为“对特征的条件残差”，而不是“把原 backbone 改写成另一个网络”。**

这就是 zero-conv style injection 有效的原因：

- `ZeroConv` 在初始化时输出为 0，意味着模型起点就是原始 AnySplat
- 风格条件通过加性残差逐步进入特征流，不会一开始就冲垮几何
- 风格注入既可以放在 Aggregator 中间层，也可以只放在 Head 前，因此对不同 transformer 骨干都相对通用

更重要的是，作者把文本与图像统一放入 Long-CLIP 共享空间中。这样风格不再是“只能由 style image 指定”的封闭接口，而是一个可插值、可语义微调的连续条件空间。

### 为什么这个设计有效？

- 冻结 backbone 保住相机、深度与 3D 点分布，减少风格训练对几何的污染
- style branch 只学习外观相关变化，等价于把训练难点从“重建 + 风格联合学习”降为“对稳定几何做条件偏移”
- patch-level CLIP directional loss 弥补了全局 CLIP 只看语义、难约束局部纹理的问题

### 战略权衡

- **优点**：轻量、快、易迁移、支持文本和图像统一条件
- **局限**：更像“全局风格/材质控制器”，不是对象级精细编辑器；同时它依赖 backbone 的原始几何质量

## Part III / Technical Deep Dive

### Pipeline

```text
unposed multi-view images + text/image style condition
-> frozen AnySplat backbone predicts cameras and geometry
-> copied Aggregator / Gaussian Head form style branch
-> Long-CLIP encodes style input into shared embedding
-> zero-conv injectors add style residuals at selected layers
-> Gaussian Adapter merges frozen geometry with stylized appearance
-> render stylized multi-view-consistent 3DGS
```

### 关键模块

#### 1. 多模态风格条件

作者使用 Long-CLIP 编码文本或风格图，并在训练时交替使用 text-conditioned 与 image-conditioned batch，以缓解共享语义空间中残留的模态差异。

#### 2. Zero-Convolution Style Injection

相比直接改 `Q/K/V` 或重训整套 attention，AnyStyle 选择更“外插式”的加法注入：

- 先把风格嵌入投影到 token 维度
- 再经 zero-initialized `1x1 conv` 形成残差
- 残差加到中间层 token 或 head 输入 token 上

这个设计使得即便只在 head 前注入风格，仍能取得很强结果，说明该风格接口本身已经足够有效。

#### 3. 训练目标

损失由四部分组成：

- `VGG content loss`：保结构
- `VGG style loss`：保纹理统计
- `global CLIP directional loss`：保整体语义风格方向
- `patch CLIP directional loss`：保局部纹理和局部风格一致性

### 关键实验信号

- `AnyStyle_img` 在四个测试场景中取得最优 `ArtFID`，优于 `Styl3R` 与 `Stylos`
- `AnyStyle_txt` 在四个场景中取得最优 `ArtScore`，说明文本条件不只是能跑通，而是真正带来了可读的风格控制
- 用户研究包含 `40` 位参与者、`20` 个案例、每题 `800` 份回答，结果对 AnyStyle 的偏好具有统计显著性
- ablation 显示：只做 `Head injection` 也很强，但 full model 的 FID 更低；移除 style loss 或 CLIP loss 都会明显退化；若放开所有几何特征一起更新，会引入可见几何伪影与视角不一致

### 对当前 idea 的启发

这篇论文对你当前方向最有价值的点，不在“4D”而在 **feed-forward 外观编辑接口**：

- 它证明了“冻结几何主干 + 条件残差编辑头”这条路是可行的
- 文本/图像共享条件空间，很适合作为未来 4D 编辑里 appearance/material branch 的设计参考
- head-only injection 的有效性说明，外观编辑未必要深度侵入 backbone，这对你后续控制复杂度很有帮助

### 实现约束

- 重建骨干基于 `AnySplat`
- 内容监督使用 `DL3DV-480P`，风格监督使用 `WikiArt`
- 文本风格描述由 `MiniCPM-V4.5` 生成

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/arXiv_2026/2026_AnyStyle_Single_Pass_Multimodal_Stylization_for_3D_Gaussian_Splatting.pdf]]
