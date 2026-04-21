---
title: "Robust Multi-Object 4D Generation for In-the-wild Videos"
venue: CVPR
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 4d-generation
  - multi-object
  - occlusion-aware
  - joint-splatting
  - object-centric-SDS
  - point-tracking
  - monocular-video
  - status/analyzed
core_operator: 先把单目多对象视频分解为对象级高斯，再在全局空间做 depth-guided early composition，通过 multi-object joint splatting 建模跨对象遮挡，同时配合 transformed object-centric SDS、instance mask rendering 与时间变化 SH 调整，在一个统一优化框架内兼顾 observed-frame 拟合与 unseen-view 生成。
primary_logic: |
  输入单目多对象视频后，先依靠分割与跟踪抽取各对象轨迹并为每个对象初始化 3D Gaussians，
  再利用深度估计把对象组合到统一全局坐标系中，
  通过 joint Gaussian splatting 优化 observed frames 的 RGB / flow / instance consistency，
  同时把对象高斯变换到 object-centric 视角上施加 SDS 生成约束，
  最终生成能够处理重遮挡、多主体交互和 novel view 的 4D 场景表示。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_Robust_Multi_Object_4D_Generation_for_In_the_wild_Videos.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T20:35
updated: 2026-04-18T20:35
---

# Robust Multi-Object 4D Generation for In-the-wild Videos

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chu_Robust_Multi-Object_4D_Generation_for_In-the-wild_Videos_CVPR_2025_paper.pdf) · [Project Page](https://genmojo.github.io/)
> - **Summary**: 这篇工作提出 GenMOJO，核心是把“多对象 + 重遮挡 + 单目视频”的 4D 生成问题，从 DreamScene4D 那种对象独立优化、最后再组合的 late-composition 路线，改成先在全局空间 early composition，再做 multi-object joint splatting。这样优化阶段就显式感知到遮挡关系，而不是事后靠深度排序补救。
> - **Key Performance**:
>   - 在 MOSE 上，GenMOJO 达到 `85.41 CLIP / 25.56 PSNR / 0.168 LPIPS`，用户偏好 `63.2%`，优于 DreamScene4D。
>   - 点跟踪评测上，MOSE 达到 `ATE 9.91 / MTE 8.47 / A-EPE 15.46 / M-EPE 11.11`，显著优于未用 point-track supervision 的方法。
>   - Ablation 中去掉 multi-object joint splatting 后，MOSE 上 `A-EPE` 从 `15.46` 恶化到 `23.48`，`PSNR` 从 `25.56` 掉到 `22.88`。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇解决的是：

**如何从单目、多对象、重遮挡的真实视频里，生成完整的 4D 场景，而不是只重建可见表面。**

问题难点比常规 4DGS 更复杂：

- 多个对象彼此遮挡
- 很多视角从未被观察到
- 单个对象的运动与对象间遮挡纠缠在一起
- 仅靠 object-centric priors 容易忽略跨对象交互

所以作者不满足于“每个对象都生成得像”，而是想让整个场景在时空上都 coherent。

### 它的方法直觉

GenMOJO 的方法直觉非常清晰：

**遮挡必须在优化时被建模，而不能等每个对象都优化完了再靠后处理拼起来。**

因此它采用：

- 对象分解，但不是对象独立终结
- 全局 early composition，而不是 late composition
- joint splatting，使 observed-frame loss 从一开始就看见跨对象 occlusion

### 一句话能力画像

- **输入**：monocular multi-object video
- **核心难点**：heavy occlusion + unobserved views
- **表示**：per-object deformable Gaussians in global scene
- **生成约束**：object-centric SDS
- **附加能力**：可从 4D 表示中直接抽 point tracks

### 对我当前方向的价值

这篇对你很重要的地方在于，它不是单纯的“生成式 4DGS”，而是：

**给出了多对象 4DGS 在遮挡条件下该如何组织优化图。**

这对你当前方向的价值主要在：

- 对 `4DGS_Editing`：如果未来要做多对象编辑、插入/替换、遮挡感知编辑，joint splatting 和 instance-aware rendering 是很强的参考。
- 对 `4DGS_Reconstruction`：即便你不做生成，early composition + occlusion-aware optimization 也值得借鉴。
- 对 foundation 方向：它说明生成式 prior 真正好用的前提，是和全局场景约束结合，而不是独立对象各自 hallucinate。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

核心创新点有四个：

- **early composition**：先把对象放到同一全局坐标系，再优化动态。
- **multi-object joint splatting**：训练时显式建模跨对象 occlusion。
- **instance mask rendering loss**：避免 Gaussian 在遮挡和交互时“跳”到别的对象上。
- **transformed object-centric SDS**：通过可微 affine 变换，把全局场景里的对象映射到 object-centric 视角里，再用 object prior 做 unseen-view regularization。

这比 DreamScene4D 更强的根本原因，不是 diffusion prior 更强，而是优化图更对。

### The "Aha!" Moment

这篇最关键的 aha 是：

**多对象场景的问题不在于“每个对象不够好”，而在于对象之间的相对关系没有在优化时被约束。**

一旦你把对象拆开独立优化，再在最后阶段重组：

- 遮挡只能靠事后排序
- 交互无法反向约束运动
- point tracks 在遮挡段最容易漂

GenMOJO 通过 early composition 把这些关系提前放进了 loss 里。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**：关系非常近，尤其是多对象编辑。它提供了“对象级可控表示 + 遮挡感知联合渲染 + 实例身份约束”的完整框架。
- **与 3DGS_Editing**：静态编辑里也有对象身份漂移问题，instance mask rendering 的思路可迁移到多实例静态场景编辑。
- **与 feed-forward Gaussians**：这篇仍是 test-time optimization，不是前馈生成。但它定义了 feed-forward multi-object 4D model 很可能需要具备的三个结构先验：对象分解、全局组合、遮挡感知。

### 局限、风险、可迁移点

- **局限**：依赖高质量分割、跟踪、深度初始化和 object-centric diffusion prior。
- **风险**：若对象掩码或深度估计抖动，global composition 仍可能沿深度轴跳动。
- **可迁移点**：
  - early composition instead of late composition
  - joint splatting for occlusion-aware optimization
  - instance identity as rendered attribute
  - 把 4D scene optimization 和 point motion evaluation 绑定起来

---

## Part III / Technical Deep Dive

### Pipeline

```text
monocular multi-object video
-> segmentation and tracking
-> initialize 3D Gaussians per object
-> depth-guided composition into global scene
-> joint splatting with RGB / flow / instance losses
-> transform each object to object-centric frames
-> apply SDS for unseen viewpoints
-> optimize dynamic Gaussians and extract point tracks
```

### 关键模块

#### 1. Multi-object Joint Splatting

这是全文的核心 operator。  
所有对象的高斯一起渲染、一起反传，因此 observed frame 中的遮挡关系会直接约束 4D 高斯运动，而不是事后排序。

#### 2. Instance and Occlusion-aware Rendering

作者给每个 Gaussian 固定 one-hot instance label，并渲染 instance segmentation map。  
这样优化如果想降低 instance loss，只能通过移动 Gaussian，而不能靠“身份漂移”糊过去。

#### 3. Transformed Object-centric SDS

全局场景优化和 object prior 之间天然坐标系不一致。  
GenMOJO 用可微 affine 变换把对象送到 object-centric camera 中渲染，再对接 SDS，这一步是把 generative prior 真正接进场景优化图的关键。

#### 4. Time-varying SH Adjustment

作者还允许颜色随时间平滑变化，以适配真实视频里的光照和阴影变化。  
这让方法比“颜色固定”的动态高斯方法更适合 in-the-wild 视频。

### 关键实验信号

- Novel view 合成上，MOSE / DAVIS 两个复杂真实数据集都优于 DreamScene4D、Consistent4D、DreamGaussian4D。
- Point tracking 上，MOSE 的优势尤其明显，说明 joint motion optimization 真正在困难遮挡段带来结构收益。
- Ablation 明确证明 multi-object joint splatting 是最关键组件，去掉以后无论 view synthesis 还是 motion metrics 都显著退化。

### 对当前研究最可迁移的 operator

- **early-composed multi-object global scene**
- **joint splatting as occlusion-aware supervision**
- **instance-mask rendering to stop identity switch**
- **object-centric SDS inside a global scene optimizer**

如果你后面重点转向 `4DGS_Editing`，这篇会是多对象遮挡场景下非常值得反复引用的结构性论文。

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_Robust_Multi_Object_4D_Generation_for_In_the_wild_Videos.pdf]]
