---
title: "4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models"
venue: CVPR
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 4dgs
  - 4d-language-field
  - mllm-captioning
  - object-wise-video-prompting
  - status-deformable-network
  - open-vocabulary-query
  - time-sensitive-query
  - status/analyzed
core_operator: 不再直接把 CLIP 视频特征硬 lift 到 4DGS，而是先用 MLLM 生成 object-wise 时序 captions，再把文本嵌入作为像素对齐监督学习 4D 语言场，并用 status deformable network 约束语义状态随时间平滑过渡。
primary_logic: |
  先用 SAM 与跟踪得到跨帧对象掩码，再通过 multimodal object-wise video prompting 让 MLLM 为每个对象生成时序 captions，
  然后将 captions 编码成共享语言空间中的句向量并作为 time-varying semantic supervision，
  同时用 status deformable network 将每个 Gaussian 的动态语义限制为若干状态原型的线性组合，
  最终联合 time-agnostic 与 time-sensitive 语义场完成开放词汇 4D 查询。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_4D_LangSplat_4D_Language_Gaussian_Splatting_via_Multimodal_Large_Language_Models.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-09T18:42
updated: 2026-04-09T18:42
---

# 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://4d-langsplat.github.io/) · [arXiv 2503.10437](https://arxiv.org/abs/2503.10437)
> - **Summary**: 4D LangSplat 想解决的是“动态场景里的开放词汇语义查询”问题。它的核心突破并不是更强的视觉 backbone，而是绕开了 CLIP 在视频语义上的天然短板，转而让 MLLM 先把对象随时间的状态变化写成文字，再把这些文字嵌回 4D 语言场。
> - **Key Performance**:
>   - 在 HyperNeRF 的 time-sensitive querying 上，平均达到 `90.83% Acc / 72.26% vIoU`。
>   - 在 time-agnostic querying 上，达到 `82.48% mIoU`（HyperNeRF）与 `85.11% mIoU`（Neu3D）。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

静态 3D language field 已经可以做开放词汇查询，但到了动态 4D 场景后，问题马上变难：

- 物体会移动、变形、切换状态
- 用户既可能问“狗在哪里”，也可能问“正在奔跑的狗在哪里”

前者是**time-agnostic** 查询，后者是**time-sensitive** 查询。  
而 CLIP 这类静态图文模型并不擅长表达“动作、状态、时序变化”。

### 核心能力定义

- **输入**：动态视频场景及其 4D Gaussian 表示
- **输出**：支持时间无关与时间相关语义检索的 4D language field
- **特别能力**：支持 open-vocabulary query，并能区分对象“何时处于某种状态”

### 真正的挑战来源

- 4D 语义需要对象级、像素对齐、跨时间一致的 supervision
- 纯视觉特征很难精准描述状态变化
- 如果每个 Gaussian 的语义可以任意跳变，时间一致性会很差

### 边界条件

- 方法依赖对象分割与跟踪质量
- MLLM caption 质量会直接影响语义监督质量
- “状态原型”假设更适合有限状态平滑转移，不是无限复杂语义变化的完全表达

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

这篇论文最聪明的一点，是把“动态语义建模”从视觉空间改写到了语言空间：

- 不强迫视觉模型直接输出高质量 4D 语义特征
- 先把视频中对象的时序行为描述成文字
- 再利用共享文本嵌入空间做 4D 查询

这其实是把“语义监督难题”从 feature engineering 变成了 caption generation + sentence embedding。

### The "Aha!" Moment

真正的 aha 是：

**CLIP 不是不会做语义，而是不适合表达动态对象的时间状态；如果要建 4D 语言场，不如先让 MLLM 把对象随时间的状态说清楚。**

也就是说，4D LangSplat 不是在问“如何得到更好的视频视觉特征”，而是在问：

**能不能把视频先翻译成对象级时序语言，再把语言重新落到 4D 场中？**

### 为什么这个设计有效

作者通过 multimodal object-wise video prompting 让 MLLM 生成：

- 视频级动作描述
- 帧级对象状态 caption

这样得到的 supervision 具备三种以前很难同时拿到的性质：

- 对象级
- 像素对齐
- 时间语义明确

随后，status deformable network 又进一步加入结构先验：  
每个 Gaussian 的语义不是任意漂移，而是在 `K` 个状态原型之间平滑插值。

### 战略权衡

- 优点：time-sensitive query 能力显著增强，语义场更贴近真实语言表达
- 代价：系统明显更依赖 MLLM、分割与文本嵌入质量

## Part III / Technical Deep Dive

### Pipeline

```text
video
-> SAM + tracking for object masks
-> multimodal object-wise video prompting
-> MLLM-generated video-level and frame-level captions
-> sentence embeddings as pixel-aligned supervision
-> 4D language field + status deformable network
-> time-agnostic and time-sensitive open-vocabulary querying
```

### 关键模块

#### 1. Multimodal Object-Wise Video Prompting

作者不是直接把整段视频丢给 MLLM，而是先通过：

- contour highlight
- gray background
- blur background

把目标对象视觉上“提纯”，再让 MLLM 先总结整段视频里的对象运动，再生成每帧对象 caption。  
这一步本质上是在给 MLLM 造更稳的 object-centric 输入接口。

#### 2. Text Embeddings As 4D Supervision

每个对象每一帧的 caption 会被编码成句向量，然后将该对象 mask 内像素统一赋成对应 embedding。  
这就把自由文本查询和 4D 场中的像素监督接到了同一个语言空间里。

#### 3. Status Deformable Network

这是论文最值得保留的 operator 之一。  
作者假设每个 Gaussian 在时间上只会在有限个语义状态间平滑切换，因此令其语义特征由 `K` 个状态原型线性组合而成，再用 MLP 从时空特征预测状态权重。

这个设计相比自由形变语义场更稳，也更符合真实动作状态变化的先验。

#### 4. Query Strategy

- **time-agnostic query**：先用静态语义场定位对象
- **time-sensitive query**：再在候选区域里用时间敏感语义场筛出真正相关时间段

这其实相当于把“对象是谁”和“对象何时处于某状态”分两层来做。

### 关键实验信号

- 与 LangSplat、Feature-3DGS、Gaussian Grouping 等相比，time-agnostic query 精度更高
- 与 Deformable CLIP、Non-Status Field 相比，time-sensitive query 的优势更明显
- ablation 表明：visual prompt、text prompt、状态数 K 都会直接影响查询质量

### 少量关键数字

- Time-sensitive querying 平均：`90.83% Acc / 72.26% vIoU`
- Time-agnostic querying：`82.48% mIoU`（HyperNeRF），`85.11% mIoU`（Neu3D）
- 在 `chick chicken` 场景上，`K=3` 的状态原型设置表现最好

### 对你可迁移的 operator

- **MLLM-generated object-wise captions as semantic supervision**
- **language-first instead of vision-first dynamic semantics**
- **state-prototype constrained semantic deformation**
- **decoupling time-agnostic and time-sensitive querying**

如果你后面做 4DGS 编辑或语义控制，这篇论文非常值得参考，因为它提供了一个很强的思路：  
**别只想着把 2D 视觉特征 lift 到 4D，也可以把“语言本身”先 lift 到 4D。**

### 实现约束

- 使用 `Qwen2-VL-7B` 生成时序 captions
- 使用 `e5-mistral-7b` 做 sentence embedding
- 实验在单张 `Nvidia A100` 上完成

## Local Reading / PDF 参考

![[paperPDFs/Gaussian_Splatting_Foundation/CVPR_2025/2025_4D_LangSplat_4D_Language_Gaussian_Splatting_via_Multimodal_Large_Language_Models.pdf]]
