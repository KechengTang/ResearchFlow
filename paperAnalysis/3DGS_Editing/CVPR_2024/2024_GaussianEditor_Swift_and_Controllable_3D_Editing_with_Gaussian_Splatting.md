---
title: "GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting"
venue: CVPR
year: 2024
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - scene-editing
  - semantic-tracing
  - hierarchical-gaussian-splatting
  - localized-editing
  - 3d-inpainting
  - diffusion-guidance
  - status/analyzed
core_operator: 用 Gaussian semantic tracing 在训练过程中持续追踪待编辑高斯，再用 HGS 以代际锚定方式稳定随机扩散指导下的 3DGS 编辑，并补上面向增删物体的 3D inpainting 流程。
primary_logic: |
  先给 3D Gaussians 赋予可继承的语义标签并在 densification 中持续追踪目标区域，
  再只对目标高斯施加编辑梯度，同时用 HGS 约束旧高斯、释放新高斯去承接细节，
  最后结合 2D inpainting 与 mesh-to-Gaussian 转换实现物体删除与加入。
pdf_ref: paperPDFs/3DGS_Editing/CVPR_2024/2024_GaussianEditor_Swift_and_Controllable_3D_Editing_with_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-09T18:42
updated: 2026-04-09T18:42
---

# GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GaussianEditor_Swift_and_Controllable_3D_Editing_with_Gaussian_Splatting_CVPR_2024_paper.html) · [Project Page](https://buaacyw.github.io/gaussian-editor/) · [PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_GaussianEditor_Swift_and_Controllable_3D_Editing_with_Gaussian_Splatting_CVPR_2024_paper.pdf)
> - **Summary**: 这篇工作是较早把 3D Gaussian Splatting 真正拉进“可控 3D 编辑”场景的代表作，核心不是单纯接一个 2D diffusion 做 SDS，而是围绕“怎么持续锁定该改的高斯、怎么在强随机指导下稳住结构、怎么补齐增删物体流程”补了一整套工程化机制。
> - **Key Performance**:
>   - 单次编辑通常只需 `5-10 分钟`，显著强调了 3DGS 相比 NeRF 编辑流程的速度优势。
>   - 论文给出约 `1 秒` 的 3D Gaussian 语义标注、约 `2 分钟` 的物体移除、约 `5 分钟` 的物体加入流程。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？
GaussianEditor 针对的是这样一个实际痛点：  
**我们想做文本驱动或局部指定的 3D 场景编辑，但既要快，又要能只改目标区域，还要支持删除/加入物体这类显式操作。**

NeRF 系编辑方法虽然可扩展，但训练慢、局部可控性差；传统 mesh/point cloud 编辑又很难自然表达复杂外观。  
3DGS 看起来很适合编辑，因为它显式、渲染快，但它也有两个新问题：

- 很难持续准确定位“应该被改的高斯”
- 在扩散模型这种高随机指导下，高斯参数会直接被冲得很不稳定

### 核心能力定义

- **输入**：一个已重建的 3DGS 场景，加上文本指令，或配合局部掩码/框选约束
- **输出**：满足编辑要求、且尽量不破坏非目标区域的 3D Gaussian 场景
- **特别能力**：不仅做风格和外观修改，还支持物体移除与新物体加入

### 真正的挑战来源

- 3D 表示在训练中会持续变化，静态 2D/3D mask 很快失效
- GS 是显式粒子表示，没有 MLP 那种“缓冲层”，随机扩散梯度更容易直接把几何与纹理冲散
- 加/删物体不是简单局部重着色，而是需要处理结构补洞、边界过渡、坐标对齐

### 边界条件

- 方法依赖已有 3DGS 重建质量
- 语义 tracing 需要先有可用的 2D 分割结果做反投影
- HGS 更像“稳定编辑训练”的结构性补丁，不是通用几何先验，极大结构改动依然困难

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

这篇论文最重要的地方，是把 3DGS 编辑拆成了三个层次的问题：

- **目标定位**：到底哪些 Gaussians 应该被改
- **训练稳定性**：扩散指导下，哪些 Gaussians 应该更稳，哪些可以更自由
- **编辑操作完备性**：不仅要“改”，还要能“删”和“加”

它不是把 InstructPix2Pix 或 SDS 直接往 3DGS 上套，而是围绕 3DGS 这个显式表示重新补齐编辑基础设施。

### The "Aha!" Moment

真正的 aha 在于：

**3DGS 编辑失败，很多时候不是 2D diffusion 不够强，而是 3D 表示层没有“持续可追踪的目标集合”和“分代稳定机制”。**

GaussianEditor 的两个关键补丁正好对应这两个缺口：

- `Gaussian semantic tracing` 让“该改谁”在训练全过程里一直可追踪
- `Hierarchical Gaussian Splatting (HGS)` 让旧高斯逐渐冻结，新高斯负责吸收随机指导带来的细节变化

### 为什么这个设计有效

Gaussian semantic tracing 的本质，是把静态 mask 升级成**跟着优化过程一起演化的 3D 动态语义掩码**。  
因为 densification 后的新高斯会继承父高斯标签，所以“编辑对象集合”不会在训练中丢失。

HGS 的本质，则是把高斯按 densification 代际分层：

- 老代高斯承担结构与主体稳定性
- 新代高斯承担细节吸收与局部修补

这相当于在原本过于“流体化”的 3DGS 上，人工造出一个从粗到细、从稳到活的层级优化秩序。

### 战略权衡

- 优点：编辑更稳、更可控，也终于能合理支持 3D inpainting
- 代价：系统流程明显更复杂，依赖分割、反投影、anchor loss、mesh 转换等多组件协同

## Part III / Technical Deep Dive

### Pipeline

```text
text / local edit instruction
-> multi-view renderings + 2D segmentation
-> Gaussian semantic tracing
-> target-only Gaussian updates
-> HGS with generation-wise anchor constraints
-> optional 3D inpainting branch for object removal / insertion
-> edited 3D Gaussian scene
```

### 关键模块

#### 1. Gaussian Semantic Tracing

作者为每个 Gaussian 增加语义标签，并通过多视角渲染 + 2D 分割结果反投影回 3D。  
关键点不是“一次性标完”，而是**在 densification 时让新点继承父点语义**，从而持续追踪编辑目标。

#### 2. Hierarchical Gaussian Splatting

HGS 将高斯按生成轮次分代，并对不同代施加不同强度的 anchor loss。  
随着训练推进，旧代高斯越来越稳，新代高斯更自由地吸收编辑细节，这显著缓解了 SDS 类随机指导导致的高斯扩散、糊化和场景整体漂移。

#### 3. 3D Inpainting

- **删除物体**：先删目标高斯，再对边界区域做 mask 精修，并用 2D inpainting 图像监督修补
- **加入物体**：先在单视角做 2D inpainting，再把前景转换为粗 3D mesh，转成 3DGS 后并入原场景并继续细化

这一步很重要，因为它把“3DGS 编辑”从纯 appearance editing 拉到了更完整的 scene manipulation。

### 关键实验信号

- 论文反复强调的不是某个单一指标，而是三件事同时成立：`速度`、`局部可控`、`多样编辑能力`
- Qualitative 结果显示它能稳定做到只改目标脸部、人物或场景局部，而非整幅场景一起飘
- HGS 的 ablation 很关键：没有 HGS 时，目标区域容易过度 densify，背景也更容易被误伤

### 少量关键数字

- 单次编辑优化一般为 `500-1000 steps`
- 编辑总耗时通常约 `5-10 分钟`
- 3D 语义 tracing 本身大约 `1 秒`

### 对你可迁移的 operator

- **dynamic 3D target tracing instead of static masks**
- **generation-wise stabilization for GS editing**
- **3D inpainting as a first-class editing primitive**

如果你后面做 3DGS 或 4DGS 编辑系统，这篇论文最值得记住的不是某个 diffusion backbone，而是：  
**要先把“编辑对象追踪”和“表示层稳定机制”设计好。**

### 实现约束

- 原始实验在单张 `RTX A6000` 上完成
- 编辑过程依赖已有多视角重建与若干视角采样
- 物体加入流程还依赖 2D inpainting 与 image-to-3D 辅助模块

## Local Reading / PDF 参考

![[paperPDFs/3DGS_Editing/CVPR_2024/2024_GaussianEditor_Swift_and_Controllable_3D_Editing_with_Gaussian_Splatting.pdf]]
