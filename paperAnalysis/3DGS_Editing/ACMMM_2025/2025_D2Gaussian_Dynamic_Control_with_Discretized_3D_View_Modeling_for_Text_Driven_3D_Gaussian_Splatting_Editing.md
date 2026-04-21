---
title: "D2Gaussian: Dynamic Control with Discretized 3D View Modeling for Text-Driven 3D Gaussian Splatting Editing"
venue: ACMMM
year: 2025
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - scene-editing
  - text-guided-editing
  - multi-view-consistency
  - controlnet
  - dynamic-controller
  - status/analyzed
core_operator: 把连续 3D 视角编码成离散 view tokens，并用这些 token 动态组装多视图一致的扩散编辑流水线，让文本驱动 3DGS 编辑不再对所有视图使用同一条静态指令。
primary_logic: |
  输入已有 3DGS 场景与相机参数，先用 codebook 和 feature mapper 把连续 3D 视角离散成 view token，
  再由 dynamic controller 按编辑目标动态追加控制条件并微调扩散式 2D 编辑模型，
  最后用 multi-view consistent iterative editing、MVS-Refine 和 3D-CLIP-SIM 迭代优化，得到更稳定的多视图 3D 编辑结果。
pdf_ref: paperPDFs/3DGS_Editing/ACMMM_2025/2025_D2Gaussian_Dynamic_Control_with_Discretized_3D_View_Modeling_for_Text_Driven_3D_Gaussian_Splatting_Editing.pdf
category: 3DGS_Editing
created: 2026-04-18T14:52
updated: 2026-04-18T14:52
---

# D2Gaussian: Dynamic Control with Discretized 3D View Modeling for Text-Driven 3D Gaussian Splatting Editing

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [DOI](https://doi.org/10.1145/3746027.3754728)
> - **Summary**: D2Gaussian 很值得补进你的库，因为它抓住了一个 3DGS 文本编辑里经常被低估的问题: 不同视角其实不应该被同一句文本、同一条 2D 编辑链路平等处理。它通过离散 view token 和动态控制流水线，把“视角敏感的多视图编辑”显式建模了。
> - **Key Performance**:
>   - 论文明确报告其多视图一致性和视觉结果优于已有 SOTA。
>   - 额外构建了 `3D-MagicBrush` 基准和 `3D-CLIP-SIM` 指标，说明作者不只是在提方法，也在补评测基础设施。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇论文瞄准的是 3DGS 文本编辑里两个连在一起的问题：

- 多视图上下文一致性不够
- 单视图里文本和图像的跨模态一致性也不够

很多已有方法会把 3D 表示渲染成多视图图片，然后直接把同一条 prompt 套到所有视角上，再调用冻结的 2D 编辑模型。这样做的问题是：

- 忽略了不同相机视角在文本空间里的差异
- 单一 2D 编辑器无法同时兼顾风格化和局部语义编辑
- 最终容易出现多视角不一致和明显伪影

因此 D2Gaussian 的核心任务并不是简单“让文本能改 3DGS”，而是：
**让 3D 视角信息真正进入编辑控制链路。**

## Part II / High-Dimensional Insight

### 方法真正新在哪里

它真正新的点有两层：

1. 不再把连续相机视角当作隐含条件，而是离散成可学习的 view tokens
2. 不再依赖固定编辑模型和固定控制条件，而是按任务动态构造 editing pipeline

这两层结合起来，使它不像 GaussCtrl 那样主要强调一致性约束，也不像普通 InstructPix2Pix 迁移那样只做单模型编辑，而是更像一个 **view-aware editing orchestrator**。

### The "Aha!" Moment

真正的 aha 在于：

**多视图编辑失败，往往不是因为模型不会编辑，而是因为它没有显式表示“这个视角看到的内容和那个视角不一样”。**

D2Gaussian 的解决方式是：

- 用 codebook 把连续视角离散化
- 用这些离散 token 去引导扩散模型
- 让控制条件随任务动态切换，而不是固定一条链

这意味着它在处理 3DGS 编辑时，已经开始把“视角建模”当作一等公民，而不是纯后处理一致性问题。

对你来说，这一点很重要，因为 4DGS 编辑里同样会碰到“视角 + 时间”联合控制的问题。D2Gaussian 虽然是 3DGS 论文，但它提供了一个很强的启发：
**离散化控制 token 也许比直接硬塞连续条件更适合复杂编辑流水线。**

### 权衡与局限

- 优势：多视图建模更显式，控制链更灵活，评测也更完整
- 局限：本质仍是 image-editing-driven 3DGS editing，对原生 3D 编辑接口依赖较弱

## Part III / Technical Deep Dive

### Pipeline

```text
3DGS scene + camera parameters + text prompt
-> discretized 3D view modeling with codebook
-> dynamic controller selects and adds control conditions
-> diffusion-based multi-view consistent editing
-> MVS-Refine + 3D-CLIP-SIM iterative optimization
-> edited 3DGS scene
```

### 关键技术点

#### 1. Discretized 3D View Modeling

论文构造 codebook 和 feature mapper，把连续视角编码成离散 token。这样每个视角不再只是相机参数，而是有了可被扩散模型利用的“视角语义索引”。

#### 2. Dynamic Multi-View Editing Pipeline

作者明确指出单一编辑模型只擅长某一类任务，所以他们用 dynamic controller 动态组装控制条件和编辑链路，而不是依赖一套冻结范式。

#### 3. MVS-Refine and 3D-CLIP-SIM

论文不仅做方法，还专门提出多视图细化模块和新的 3D 评测指标，说明它很重视“编辑结果是否真的在 3D 上成立”。

### 关键信号

- 作者在文中把多视图一致性提升作为主结果，而不是次级收益。
- 新基准 `3D-MagicBrush` 也很值得注意，因为这说明他们意识到 3D 编辑比较缺标准化评测。

### 对你的价值

虽然它不是 4DGS 编辑论文，但它很值得作为高优先级 3DGS 编辑参考：

- 给 4DGS 编辑提供了 **离散控制 token** 的思路
- 给多视角一致性提供了 **动态流水线** 的设计角度
- 对 text-driven editing 的 benchmark 侧也有补充价值

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ACMMM_2025/2025_D2Gaussian_Dynamic_Control_with_Discretized_3D_View_Modeling_for_Text_Driven_3D_Gaussian_Splatting_Editing.pdf]]
