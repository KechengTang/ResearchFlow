---
title: "RoMaP: Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling"
venue: ICCV
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - part-level-editing
  - local-editing
  - 3d-masking
  - score-distillation
  - regularized-sds
  - soft-label-segmentation
  - status/analyzed
core_operator: 先用 3D-GALP 把 part segmentation 提升为视角相关的 3D Gaussian soft-label 分割，再用包含 prior removal、robust mask 与 SLaMP anchored L1 的 regularized SDS 精确驱动局部高斯编辑。
primary_logic: |
  输入高斯场、部件分割提示与编辑提示，
  先通过注意力伪分割与 3D-GALP 在高斯层建立可跨视角稳定的 part-level 3D mask，
  再对目标区域做 Gaussian prior removal，并用 SLaMP 生成的 part-edited 2D 图像提供 anchored L1 方向，
  最后以 regularized SDS 只更新目标局部高斯，实现精确且可大幅偏离原语义先验的部件编辑。
pdf_ref: paperPDFs/3DGS_Editing/ICCV_2025/2025_RoMaP_Robust_3D_Masked_Part_level_Editing_in_3D_Gaussian_Splatting_with_Regularized_Score_Distillation_Sampling.pdf
category: 3DGS_Editing
created: 2026-04-18T16:25
updated: 2026-04-18T16:25
---

# RoMaP: Robust 3D-Masked Part-level Editing in 3D Gaussian Splatting with Regularized Score Distillation Sampling

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_Robust_3D-Masked_Part-level_Editing_in_3D_Gaussian_Splatting_with_Regularized_ICCV_2025_paper.pdf) · [Project Page](https://janeyeon.github.io/romap/)
> - **Summary**: RoMaP 把 3DGS 编辑从“实例级改物体”推进到“部件级改鼻子、眼睛、头发”等更细粒度控制，关键在于同时解决了两个旧方法一直没解好的问题：part mask 不稳，以及 SDS 无法把局部编辑明确推向罕见目标。
> - **Key Performance**:
>   - 在编辑基准上取得 `CLIP 0.277 / CLIPdir 0.205 / B-VQA 0.723 / TIFA 0.674`，均优于 `DGE / GaussianEditor / GaussCtrl` 等基线。
>   - 用户研究中编辑任务得分 `0.372`，高于 `DGE 0.224`、`GaussCtrl 0.201`、`GaussianEditor 0.201`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

RoMaP 的目标非常明确：

**把 3DGS 编辑从“改整个人、整个物体”推进到“只改左眼、鼻子、嘴唇、头发的一部分”，而且还能做非常反常、低概率的局部变化。**

这类 part-level editing 对现有方法非常难，因为原有路线大多是：

- 先在多视角 2D 图像里找目标 part。
- 再把这些 2D mask 投回 3D。
- 最后用普通 SDS 或 2D 编辑结果去推 3D 优化。

这在实例级编辑还能凑合，但到了部件级，2D 分割不稳定、边界不清、不同视角 label 冲突等问题都会被放大。

### 核心能力定义

- **输入**：3D Gaussian scene / object、分割提示、编辑提示。
- **输出**：只在指定局部发生变化、其余身份与上下文尽量保持的 3DGS。
- **擅长**：眼睛异色、鼻子材质替换、头发变形、灯罩局部材质变化等细粒度编辑。
- **能力跃迁**：从 instance-level editing 迈向真正可控的 part-level editing。

### 真正的挑战来源

- part segmentation 在多视角下极不稳定，不同视角可能只看到半个 part、多个 part 混在一起，或完全漏掉。
- 高斯边界点天然具有 soft-label 属性，一个 Gaussian 可能从不同视角看属于不同 part。
- 标准 SDS 的方向太模糊，容易把改动扩散到非目标区域，或者被原始 appearance prior 拉回去，导致“该改的地方没改，对的方向走不出去”。

### 边界条件

- 这是一个更复杂的局部编辑框架，需要先完成较稳的 part localization。
- 仍然是 optimization-based editing，不是快速前向模型。
- 对极端 part 定义或特别抽象的分割提示，伪分割质量仍然会成为上限。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

RoMaP 的整体策略可以概括成一句话：

**要做部件级 3D 编辑，必须同时把“改哪里”与“往哪改”都做成显式对象。**

所以它对应引入两个核心模块：

- `3D-GALP`：解决 part-level 3D mask。
- `regularized SDS + SLaMP`：解决局部编辑方向与强度。

### The "Aha!" Moment

最关键的 aha 有两个。

第一，**不要把 Gaussian 的部件标签当 hard label，而要当 view-dependent soft label**。RoMaP 用球谐系数来表示标签随视角的变化，并根据 label softness 选 anchor，再加上邻域一致性约束来修正 part 边界。这一步本质上是在承认 Gaussian 边界本来就是混合语义，而不是强行把它离散硬分。

第二，**标准 SDS 不适合作为部件级“方向控制器”**。它能提供优化趋势，但很难把局部区域推向“蓝左眼绿右眼”或“翡翠鼻子”这种低概率编辑。因此作者引入 `SLaMP` 生成局部编辑的 2D anchor 图，再用 anchored L1 把 SDS 往目标方向钉住，同时做 `Gaussian prior removal` 把原有外观先验削弱。

### 为什么这个设计有效？

- `3D-GALP` 让 part mask 不再完全依赖 noisy 2D segmentation，尤其改善边界区域和多视角不完整观测。
- label softness-based anchor sampling 同时采样“最稳定”与“最不稳定”的高斯点，能兼顾主体区域与边界修复。
- `prior removal` 先把原 appearance prior 削弱，避免优化总是朝“更像原物体”的方向回缩。
- `SLaMP` 的 scheduled latent mixing 则给出一个足够强但仍保留全局身份的局部编辑锚点，正好弥补 SDS 的模糊性。

### 战略权衡

- **优点**：局部性和可控性都明显强于以往基于 2D 分割 + 普通 SDS 的方案。
- **代价**：模块较多，参数和调度更复杂，尤其 `t_s` 的选择会影响“保留原貌”和“放开编辑”的平衡。
- **定位**：这是一个为了 part-level controllability 而设计的高约束局部编辑框架，不是最简单的通用编辑器。

## Part III / Technical Deep Dive

### Pipeline

```text
3D Gaussian scene/object + segmentation prompt + edit prompt
-> extract attention-based pseudo 2D segmentation
-> 3D-GALP builds a view-aware 3D Gaussian mask
-> remove dominant appearance priors on target Gaussians
-> generate SLaMP-edited 2D anchors for target parts
-> optimize target Gaussians with regularized SDS + anchored L1 + masking
-> obtain part-level edited 3DGS
```

### 关键模块

#### 1. 3D-GALP

作者给每个 Gaussian 增加标签参数 `r`，并把它也表示成 SH 系数，使其可以随视角变化。这样边界点不必被硬判成单一 part，而是能保留 mixed-label 属性。随后利用基于 label softness 的 anchor sampling 和邻域一致性损失，把初始伪分割细化成更稳的 3D mask。

#### 2. Gaussian prior removal

作者明确指出，局部编辑之所以很难做大改动，一个原因是高斯本身携带的颜色/材质先验太强。prior removal 通过把这些强先验替换成中性颜色，先把“原样子”从优化里降权，给后续目标编辑留下空间。

#### 3. SLaMP

SLaMP 是这个方法里非常关键的方向控制器。它通过对 target latent 与 original latent 做时序调度混合，先放开局部区域生成新内容，再在某个 `t_s` 之后迅速增强原图约束，从而兼顾 drastic local change 与 global identity preservation。

#### 4. Regularized SDS

最终优化不是裸 SDS，而是：

- `SDS` 负责给目标编辑方向提供 generative signal。
- `anchored L1` 负责把方向钉到 SLaMP 提供的局部编辑锚点上。
- `M3D / M2D` masking 负责把梯度锁在目标区域内。

三者组合后，RoMaP 才能真正做细粒度且幅度较大的 part-level edits。

### 关键实验信号

- 定量上 RoMaP 在 `CLIP / CLIPdir / B-VQA / TIFA` 四个指标全部最优，说明它不只是 prompt 更对，连局部问答型评估也更稳。
- 用户研究进一步表明它在主观 controllability 上显著领先。
- 消融显示，从 baseline 到 `+Mask` 再到 `+Mask & L1` 再到 `Full`，指标逐步提升，说明 robust mask 与 regularized SDS 的两个部分都不可缺。

### 少量关键数字

- `CLIP`: `0.277`
- `CLIPdir`: `0.205`
- `B-VQA`: `0.723`
- `TIFA`: `0.674`
- 用户研究：`0.372`

### 实现约束

- 编辑评测使用来自 `IN2N` 与 `NeRF-Art` 的重建 Gaussian 场景，测试 `75` 条针对不同部件的 prompts。
- 消融中，baseline 到 full 的 `CLIPdir` 从 `0.060` 提升到 `0.205`。
- `t_s` 的选取通过权衡 `SSIM` 稳定与 `CLIPdir` 较高来确定。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICCV_2025/2025_RoMaP_Robust_3D_Masked_Part_level_Editing_in_3D_Gaussian_Splatting_with_Regularized_Score_Distillation_Sampling.pdf]]
