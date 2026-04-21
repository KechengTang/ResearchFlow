---
title: "InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior"
venue: ICCV
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - interactive-editing
  - cross-view-consistency
  - geometry-aware-editing
  - clip-features
  - diffusion-guidance
  - status/analyzed
core_operator: 先让用户从初始编辑结果里选一个满意的 key view，再用 CLIP-based Semantic Consistency Selection 选出语义最接近的 reference views，将其 cross-attention 反投影成 GAP3D，并通过 Attention Fusion Network 在扩散早期偏向 3D 几何约束、后期偏向 2D 细节恢复。
primary_logic: |
  输入原始 3DGS、文本编辑指令与用户选定的 key view，
  先对多视角图像做初始 diffusion 编辑并计算各视图的 CLIP 编辑方向，
  再围绕 key view 选择语义一致的 reference views 并加权反投影其 attention，构建 3D Geometry-Consistent Attention Prior，
  随后把 GAP3D 投影回各视图得到 3D-constrained attention，并在 AFN 中与原始 2D cross-attention 动态融合，
  最终获得与 key view 在几何和外观上更一致的多视角编辑结果，并回写到 3DGS。
pdf_ref: paperPDFs/3DGS_Editing/ICCV_2025/2025_InterGSEdit_Interactive_3D_Gaussian_Splatting_Editing_with_3D_Geometry_Consistent_Attention_Prior.pdf
category: 3DGS_Editing
created: 2026-04-18T18:25
updated: 2026-04-18T18:25
---

# InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wen_InterGSEdit_Interactive_3D_Gaussian_Splatting_Editing_with_3D_Geometry-Consistent_Attention_ICCV_2025_paper.pdf) · [arXiv 2506.08461](https://arxiv.org/abs/2506.08461)
> - **Summary**: InterGSEdit 的核心不是再造一个更强的文本编辑器，而是给 3DGS 编辑加上“用户挑一个最想要的视角结果，再把这个选择传播到全部视角”的交互闭环。它把模糊文本语义锚定到 key view，再用 3D attention prior 约束后续扩散，因此特别适合笑容、张嘴这类非刚性编辑。
> - **Key Performance**:
>   - 在量化对比中达到 `CLIP Similarity 0.2285 / CTIDS 0.1531 / CDC 0.8347`，三项指标都优于 `IGS2GS / GSEditor / DGE / VcEdit`。
>   - 消融表明 `CSCS` 把 `CDC` 从 `0.7647` 拉到 `0.8218`，而 `AFN` 进一步把 `CDC` 提升到 `0.8802`，同时明显减少牙齿伪影和欠编辑区域。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

InterGSEdit 主要瞄准的是现有 3DGS 编辑里一个非常具体的问题：

**当编辑对象发生非刚性变化时，不同视角往往会得到语义略有差异的 2D 编辑结果，回写到 3D 后就会出现模糊、扭曲或局部伪影。**

例如“make him smile”这类指令，不同视角里嘴角上扬幅度、牙齿是否可见，都可能被 diffusion 解释成不一样的局部结果。  
如果直接把这些不一致图像用于 3DGS 优化，最后只会得到一个折中的、发糊的 3D 场景。

### 核心能力定义

- **输入**：原始 3DGS、多视角渲染图、文本编辑指令，以及用户从初始编辑结果中选出的 key view。
- **输出**：几何更一致、且更符合用户偏好的 3D 编辑结果。
- **擅长**：面部表情、局部非刚性外观变化、需要用户细粒度“挑结果”的交互式编辑。
- **不擅长**：完全脱离 2D diffusion 的 3D-only 编辑，也不解决大范围 3D 几何拓扑变化。

### 真正的挑战来源

- 文本提示本身语义模糊，尤其难以稳定描述细节级变化。
- 不同视角的 diffusion 编辑会产生局部特征差异，3D 优化时这些差异会互相冲突。
- 如果直接用 3D attention 完全替换 2D attention，虽然更稳，但细节会损失。

### 边界条件

- 方法仍基于 diffusion 编辑框架，3DGS 只是在后面吸收一致化后的多视角结果。
- 用户必须先从候选视图里挑一个满意的 key view，因此它是“带人工选择”的高质量编辑，而不是全自动批处理。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

InterGSEdit 的思路非常清楚：

**先用用户选择把“想要哪种编辑风格”这件事钉死，再让 3D attention prior 去把这种选择传播到其他视角。**

也就是说，作者并不试图让文本自己解决所有歧义，而是承认：

- 文本只负责给出大方向；
- 用户选出来的 key view 才是最终编辑意图的具体锚点。

### The "Aha!" Moment

真正的 aha 是：

**多视角不一致的根源，不只是几何约束不够强，而是没有一个明确的“语义参照视图”来决定其他视图该像谁。**

因此作者提出两步：

1. 用 `CLIP-based Semantic Consistency Selection (CSCS)` 在编辑方向空间里找出和 key view 最一致的 reference views。
2. 把这些 reference views 的 cross-attention 反投影到 3DGS 上，形成 `GAP3D`，再重新投影给所有视图。

这样做的本质，是把“视图间求平均”改成“围绕用户偏好的视图做有偏传播”。

### 为什么这个设计有效？

- `CSCS` 不是直接比图像内容，而是比较“编辑前后 embedding 差分”和文本编辑方向的对齐程度，因此更适合筛掉语义跑偏的视图。
- `GAP3D` 让 3DGS 成为 attention 的共享存储层，能把 reference views 的编辑共识稳定传到其他视图。
- `AFN` 不做硬替换，而是动态控制 3D/2D attention 权重：前期保几何一致，后期补细节纹理。

### 战略权衡

- **优点**：很适合非刚性编辑和用户偏好敏感的场景，交互价值高。
- **局限**：额外引入 key view 选择步骤，也意味着结果质量仍受初始 diffusion 候选集影响。

## Part III / Technical Deep Dive

### Pipeline

```text
source 3DGS + text prompt
-> initial multi-view diffusion edits
-> user selects preferred key view
-> CLIP-based Semantic Consistency Selection picks weighted reference views
-> unproject reference-view cross-attention into GAP3D
-> project GAP3D to each view as 3D-constrained attention
-> AFN dynamically fuses 3D attention with 2D cross-attention
-> diffusion re-edits views with better consistency
-> optimize edited 3DGS
```

### 关键模块

#### 1. CSCS

作者先计算 key view 和其它候选视图的 CLIP 图像编辑方向，再和文本方向作对齐。  
偏差越小，说明这个视图的编辑语义越接近 key view，于是就获得更高权重。

#### 2. GAP3D

对 reference views 的 cross-attention 做加权 unprojection，得到每个 Gaussian 上的 3D attention prior。  
这个 prior 不是最终结果，而是一个“哪些位置应该被同样关注”的 3D 几何语义约束。

#### 3. AFN

`AFN` 通过门控方式融合 `Attn3D` 与 `Attn2D`。  
扩散早期给 3D attention 更高权重，确保结构先对齐；后期逐渐回到 2D attention，以恢复局部外观细节和纹理。

### 关键实验信号

- 定量表 1 中，InterGSEdit 三项指标全部超过先前代表性方法，说明它不只是主观观感更好，而是文本对齐和多视角一致性都更强。
- 论文展示的“smile / laugh / open mouth”例子里，旧方法常出现牙齿伪影或表情强度不一，而 InterGSEdit 基本能围绕 key view 保持统一。
- 没有 `CSCS` 时，多视角参考混入语义漂移的图像，会直接带来 tooth artifacts；没有 `AFN` 时，又会出现结构虽然稳但细节编辑不充分的问题。

### 对当前知识库的价值

这篇论文的价值不只是在于“又一个 attention consistency 方法”，而是它给出了一个很清晰的交互范式：

- 用用户选择的 key view 来解决文本歧义；
- 用 3D attention prior 来解决跨视角几何一致；
- 用动态 attention fusion 来平衡几何与细节。

这比单纯堆更强的 2D prompt engineering 更接近真实可用的编辑工具。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ICCV_2025/2025_InterGSEdit_Interactive_3D_Gaussian_Splatting_Editing_with_3D_Geometry_Consistent_Attention_Prior.pdf]]
