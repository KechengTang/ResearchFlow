---
title: "View-Consistent 3D Editing with Gaussian Splatting"
venue: ECCV
year: 2024
tags:
  - 3DGS_Editing
  - gaussian-splatting
  - 3dgs
  - scene-editing
  - image-guided-editing
  - multi-view-consistency
  - cross-attention-consistency
  - iterative-editing
  - diffusion-guidance
  - status/analyzed
core_operator: 把多视角图像引导编辑拆成 cross-attention 一致性与 edited-image 一致性校正两层，并在每个 diffusion timestep 内用 3DGS 做 3D-2D 往返校准，抑制 mode collapse。
primary_logic: |
  输入源 3DGS 与文本编辑指令，
  先用 CCM 把多视角 cross-attention 逆投影到 3D Gaussians 后再渲染回各视角，统一“该改哪里”的注意区域，
  再用 ECM 把当前 edited images 蒸馏到一个临时 fine-tuned 3DGS 并重新渲染成更一致的视图，
  最后在多个 timestep 与多个 iteration 中交替细化，得到可直接监督最终 3DGS 的一致编辑图像。
pdf_ref: paperPDFs/3DGS_Editing/ECCV_2024/2024_View_Consistent_3D_Editing_with_Gaussian_Splatting.pdf
category: 3DGS_Editing
created: 2026-04-18T16:10
updated: 2026-04-18T16:10
---

# View-Consistent 3D Editing with Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [ECCV 2024 PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05210.pdf) · [Project Page](http://yuxuanw.me/vcedit/)
> - **Summary**: VcEdit 的核心不是重新发明一个 3D 编辑器，而是把现有 image-guided 3DGS 编辑里最致命的“多视角不一致”单独拿出来处理，并把 3DGS 本身变成一个跨视角一致性校准器，因此能显著减轻 flicker、语义漂移和 mode collapse。
> - **Key Performance**:
>   - 用户研究中获得 `63.93%` 偏好，明显高于 `GSEditor 34.49%` 和 `DDS 1.57%`。
>   - `CLIP directional similarity = 0.2108`，高于 `GSEditor 0.1917` 和 `DDS 0.1470`；单样本编辑通常约 `10-20 min`。

## Part I / The "Skill" Signature

### 它真正想解决什么问题？

这篇论文针对的是 image-guided 3DGS 编辑里一个非常具体但非常致命的瓶颈：

**不是 2D 编辑器不会改，而是它对每个视角各改各的，最后 3DGS 吃到的是互相冲突的监督。**

对 3DGS 来说，这种冲突比 NeRF 更危险，因为 3DGS 是显式表示，还带 densification / pruning；一旦不同视角在“同一个 3D 位置应该长成什么样”上意见不统一，模型就很容易进入 mode collapse，最后表现为局部混杂、闪烁和结构发虚。

### 核心能力定义

- **输入**：一个已重建的 3DGS 场景，加上一条文本编辑指令。
- **输出**：与目标文本更一致、且跨视角更稳定的编辑后 3DGS。
- **擅长**：人物/物体外观替换、局部属性修改、场景风格或环境级变化。
- **主要收益**：把“多视角图像编辑不一致”从一个副作用，提升成显式建模对象。

### 真正的挑战来源

- 2D diffusion 编辑器天然是单视角的，不知道不同相机其实在看同一个 3D 区域。
- 3DGS 对不一致监督很敏感，因为显式高斯会直接吸收这些冲突。
- 如果只靠最终一次 3D 重建去“平均掉差异”，往往已经太晚，信息已经相互污染。

### 边界条件

- 前提是源 3DGS 本身已经足够稳定，至少能可靠渲染出多视角图像。
- 这仍然是 image-guided editing，不是原生 3D 生成编辑；几何大改动的上限仍受 2D 编辑图像质量影响。
- 系统流程比直接套 2D 编辑器复杂，推理也不是实时。

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

VcEdit 的设计思路很清楚：

**既然 2D 编辑器缺少 3D awareness，那就不要指望它自己变得 3D-aware；而是在它前后各插一个由 3DGS 提供的“一致性校正层”。**

它没有直接重写 diffusion backbone，而是围绕现成编辑器做两个补丁：

- 在 attention 层面统一“不同视角正在关注哪一块 3D 区域”。
- 在图像结果层面统一“不同视角生成出的目标外观是否能由同一个 3D 场解释”。

### The "Aha!" Moment

真正的 aha 是：

**把 3DGS 从“被编辑对象”临时提升成“多视角共识空间”。**

具体分两步：

- `CCM`：把每个视角的 cross-attention map 逆渲染到同一组 Gaussians 上，再平均后重新渲染回 2D。这样不同视角会在“哪块 3D 区域对应 prompt 中哪个词”上达成共识。
- `ECM`：把当前 timestep 得到的 edited images 先蒸馏进一个临时 fine-tuned 3DGS，再从这个更物理一致的中间模型重新渲染视图，作为下一步编辑的更稳定输入。

这意味着 VcEdit 并不是只在最后一拍才做 3D 一致性约束，而是在 diffusion 过程内部就不断做 3D 校正。

### 为什么这个设计有效？

- `CCM` 解决的是“注意哪里”的一致性。如果不同视角对 prompt 的关注点都不一样，后面再怎么拟合都只是在拟合冲突。
- `ECM` 解决的是“改成什么样”的一致性。它把一组 still noisy 的编辑视图投回一个共享 3D 表示，再从这个共享表示重新采样出更协调的图像。
- 多 timestep 的 `Local Blend` 让未编辑区域尽量留在源图像分布里，减少全局污染。
- iteration 级别的反复细化进一步修正第一轮里没完全对齐的细节。

### 战略权衡

- **优点**：不需要重新训练编辑模型，就能显著提升 3D 编辑稳定性。
- **代价**：每个 timestep 都在做额外的 3D-2D 往返，流程比直接 image editing 更重。
- **本质定位**：这是一个“稳定高质量 image-guided 3DGS 编辑”的框架，而不是最轻量的交互式系统。

## Part III / Technical Deep Dive

### Pipeline

```text
source 3DGS + text prompt
-> multi-view renders
-> diffusion editing with CCM
-> ECM: fine-tune a copied 3DGS on edited views and re-render coherent views
-> local blend with source latents
-> repeat across timesteps
-> decode final consistent edited images
-> fine-tune source 3DGS with MAE + LPIPS
-> optional next iteration for further refinement
```

### 关键模块

#### 1. Cross-attention Consistency Module

CCM 的重点不是对像素做平均，而是对 **prompt-to-region 对齐关系** 做平均。它把各视角的 cross-attention 逆投影到 3D 高斯上，形成共享 3D attention map，再渲染回 2D 替换原 attention。这样 U-Net 在所有视角上都会把同一个文本 token 指向同一片 3D 区域。

#### 2. Editing Consistency Module

ECM 更像一个小型“物理合理性过滤器”。它把当前编辑结果先喂给一个临时 fine-tuned 3DGS，让这个中间模型吸收多视角共性、压掉互相矛盾的细节，然后再把它渲染回一致视图。这个步骤直接阻止不一致在后续 diffusion step 中累计。

#### 3. Iterative Pattern

作者把 `Gsrc -> I_src -> I_edit -> Gedit` 视为一次 iteration。前一轮的 `Gedit` 会成为下一轮的 `Gsrc`。这样后续轮次面对的是一个已经更接近目标分布的 3D 场，因此 2D 编辑器更容易给出稳定结果。

### 关键实验信号

- 最重要的定量信号不是单一视觉指标，而是 **用户偏好 + CLIP directional similarity 同时提升**，说明它既更像目标 prompt，也更符合人类主观质量判断。
- 消融里去掉一致性模块后，2D 编辑视图会明显互相冲突；只保留 `CCM` 时已明显改善，而 `CCM + ECM` 时跨视角几乎完全对齐。
- iteration 消融显示第二轮继续改善了五官、妆容和衣物等细节，说明它不是只靠第一轮“凑出来”的结果。

### 少量关键数字

- 用户研究：`63.93%`
- `CLIP directional similarity`: `0.2108`
- 单样本编辑时间：`10-20 min`

### 实现约束

- 编辑通常做 `1-3` 轮 iteration，每轮源 3DGS 优化约 `400 steps`。
- 文中实现运行在 `NVIDIA RTX A6000` 上，ECM 与最终 3DGS fine-tuning 都使用 `Adam`，学习率 `0.001`。
- 最终 3DGS 监督采用 `MAE + LPIPS`。

## Local Reading / PDF 引用

![[paperPDFs/3DGS_Editing/ECCV_2024/2024_View_Consistent_3D_Editing_with_Gaussian_Splatting.pdf]]
