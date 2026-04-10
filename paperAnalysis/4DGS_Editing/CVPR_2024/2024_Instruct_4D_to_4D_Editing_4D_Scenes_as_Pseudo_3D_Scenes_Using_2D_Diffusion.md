---
title: "Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion"
venue: CVPR
year: 2024
tags:
  - 4DGS_Editing
  - dynamic-scene-editing
  - instructpix2pix
  - pseudo-3d-editing
  - anchor-aware-attention
  - optical-flow-guided
  - temporal-consistency
  - multi-view-consistency
  - status/analyzed
core_operator: 把 4D 场景重写成 pseudo-3D 多伪视角视频编辑问题，先用 anchor-aware IP2P 与光流滑窗做时序一致编辑，再用 key pseudo-view 的深度传播和迭代数据更新把结果蒸馏回 4D 表示。
primary_logic: |
  输入 4D 场景渲染结果与文本编辑指令，先把每个固定相机视角视作一段 pseudo-view 视频，
  用 anchor-aware IP2P + optical-flow-guided sliding window 获得单个 pseudo-view 内的时序一致编辑，
  再从 key pseudo-views 向其他视角做深度传播，最后通过 iterative dataset update 持续替换训练集并优化 4D NeRF，
  得到跨视角与跨时间都更稳定的编辑后 4D 场景。
pdf_ref: paperPDFs/4DGS_Editing/CVPR_2024/2024_Instruct_4D_to_4D_Editing_4D_Scenes_as_Pseudo_3D_Scenes_Using_2D_Diffusion.pdf
category: 4DGS_Editing
created: 2026-04-10T15:20
updated: 2026-04-10T15:20
---

# Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://immortalco.github.io/Instruct-4D-to-4D/) · [CVPR 2024 Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Mou_Instruct_4D-to-4D_Editing_4D_Scenes_as_Pseudo-3D_Scenes_Using_2D_CVPR_2024_paper.html)
> - **Summary**: 这篇工作把“直接做 4D instruction-guided editing”拆成一个更可落地的问题：先把 4D scene 看成由多个 pseudo-view 视频组成的 pseudo-3D scene，先解决单个 pseudo-view 的时序一致编辑，再解决 key pseudo-view 向其他视角的传播，从而把 2D diffusion 的编辑能力蒸馏回 4D 表示。
> - **Key Performance**:
>   - 在 multi-camera `coffee martini` 风格迁移实验里，平均达到 `19.67 PSNR / 0.635 SSIM / 0.323 LPIPS-Alex`，明显优于 naive `IN2N-4D` 的 `14.11 / 0.457 / 0.512`。
>   - 消融里 full model 的 `CLIP similarity = 0.3085`，高于 `IN2N-4D` 的 `0.2790` 与 one-time pseudo-view propagation 的 `0.2876`，说明 optical flow 与 iterative pipeline 都是有效增益。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

这篇论文瞄准的是最早期、也最基础的 **instruction-guided 4D scene editing**。  
它想解决的不是“怎样让单帧被改得更像 prompt”，而是：

- 同一个动态对象在不同时间不能越改越漂
- 同一时刻的不同视角不能各改各的
- 4D 表示不能只靠拟合一堆彼此矛盾的 2D 编辑帧硬凑出结果

作者的判断很明确：直接把 2D image editing 一帧一帧套到 4D 上，会同时在 **跨时间一致性** 和 **跨视角一致性** 上出问题。

### 核心能力定义

- **输入**：4D scene 的渲染帧、文本编辑指令
- **输出**：编辑后的 4D NeRF 场景
- **能做的事**：风格迁移、属性变化、对象替换这类 instruction-guided 外观编辑
- **做不到的事**：native 4D 表示层的直接编辑、显式对象级运动建模、一次前向推理完成编辑

### 真正的挑战来源

- 4D 不是只多一个时间轴，而是对象在不同帧不再由同一组参数直接表示
- diffusion 的逐帧结果天然带多解性，直接蒸馏回 4D 很容易平均化
- 长视频 / 多视角全量编辑代价极高，不能直接把所有帧一次性送进 2D 编辑器

### 边界条件

- 方法强依赖已有 2D editor，尤其是 IP2P
- 本质上还是 test-time iterative optimization，而不是可复用的 learned editor
- backbone 用的是 4D NeRF/NeRFPlayer，不是后来更流行的 canonical Gaussian editing interface

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

这篇工作的最大价值，不是某个单独模块，而是它给出了一个很早期但非常重要的 **4D editing decomposition**：

1. 不直接编辑整个 4D scene
2. 先把 4D scene 改写成 pseudo-3D scene
3. 再把问题拆成：
   - 单个 pseudo-view 内的时序一致视频编辑
   - key pseudo-views 向其他视角的传播

这个拆法让“4D editing”第一次变得可操作。

### The "Aha!" Moment

真正的 aha 是：

**4D scene 不一定要被直接当成 4D 去编辑，也可以先重写成由多个固定相机视频组成的 pseudo-3D 场景。**

这一步的因果作用很强：

- 一旦固定相机，单个 pseudo-view 的编辑问题就更接近 video editing
- 一旦有 key pseudo-view，跨视角传播就能借用已有的 depth warping 思路
- 一旦能持续生成“更一致的 edited dataset”，4D 表示就能像 IN2N 那样做 iterative distillation

也就是说，它不是让 2D diffusion 直接“理解 4D”，而是先把 4D 改写成更适合 2D diffusion 工作的接口。

### 为什么这个设计有效

- **Anchor-aware IP2P**：把 batch 内和 batch 间的一致性显式建到 attention 里，而不是靠最终 4D 拟合兜底
- **Optical-flow sliding window**：解决长 pseudo-view 不能一次全编辑的问题
- **Key pseudo-view propagation**：避免所有视角都直接跑昂贵编辑
- **Iterative dataset update**：让编辑后的 2D 结果逐轮替换监督数据，逼近稳定 4D 解

### 战略权衡

- 优点：是最早把通用 instruction-based 4D editing 跑通的方案之一，问题拆分很清楚
- 代价：仍然重度依赖 2D 编辑器和迭代优化，编辑接口不够 native，也没有显式语义定位模块

## Part III / Technical Deep Dive

### Pipeline

```text
4D scene
-> rewrite as pseudo-3D scene with multiple pseudo-views
-> anchor-aware IP2P edits each pseudo-view in a sliding window
-> optical-flow temporal propagation within each pseudo-view
-> depth-based propagation from key pseudo-views to other views
-> iterative dataset update and 4D NeRF optimization
-> edited 4D scene
```

### 关键模块

#### 1. Anchor-aware IP2P

作者把原始 IP2P 改造成支持 batch generation 的版本，并把 self-attention 换成对 anchor frame 的 cross-attention。  
目的不是更强编辑，而是 **让多个待编辑帧共享同一风格参照**，从源头减少 batch 内 / batch 间漂移。

#### 2. Optical Flow-guided Sliding Window

这是这篇论文最关键的时序模块。  
它先用 RAFT 估计相邻帧光流，把上一段编辑结果 warp 到当前段，再让 IP2P 对 warp 后图像做 inpaint / repaint。  
因此 temporal consistency 不是靠后验平滑，而是靠“先传播再编辑”的局部递推。

#### 3. Key Pseudo-view Propagation

作者借鉴 key-view editing，只选少量 pseudo-views 真正跑昂贵编辑；其余视角靠深度投影与加权聚合补全。  
这让 4D 编辑的总成本从“所有视角全量编辑”降成“少量关键视角编辑 + 大量传播”。

#### 4. Iterative Dataset Update

这一步和 IN2N 的精神很像：  
不是一次性生成最终编辑数据，而是不断重新生成更一致的 edited dataset，再继续拟合 4D NeRF，直到收敛。

### 关键实验信号

- 定量上，`coffee martini` 的 style transfer 平均结果从 baseline 的 `14.11 PSNR / 0.457 SSIM / 0.512 LPIPS-Alex` 提升到 `19.67 / 0.635 / 0.323`。
- 在多视角风格迁移实验里，作者报告自己的方法大约 **2 小时**能得到稳定结果，而 naive `IN2N-4D` 花更久仍难收敛到一致风格。
- 消融里 full model 的 `CLIP similarity = 0.3085`，高于：
  - `IN2N-4D = 0.2790`
  - `Video Editing = 0.2631`
  - `w/o Optical Flow = 0.2792`
  - `One-time Pseudo-View Propagation = 0.2876`

### 对你当前 idea 的启发

这篇工作最值得保留的不是最终系统，而是下面三条 operator：

- **pseudo-3D decomposition**：把 4D 问题改写成更易求解的接口
- **anchor-based consistent editing**：把 consistency 从后验约束前移到编辑过程本身
- **temporal propagation before optimization**：先做传播，再做 4D 拟合

对你当前的 `semantic feed-forward 4DGS editing` 来说，它的角色更像：

- 早期 4D optimization baseline
- “为什么 4D editing 必须显式处理 temporal propagation”的直接证据
- 你后续 feed-forward temporal adapter 的历史前身

### 实现约束

- backbone 使用 `NeRFPlayer`
- 通过双 GPU 异步并行 NeRF 训练与 dataset generation 提升效率
- 依赖 `RAFT` 光流和 `IP2P` 图像编辑能力

## Local Reading / PDF 参考
![[paperPDFs/4DGS_Editing/CVPR_2024/2024_Instruct_4D_to_4D_Editing_4D_Scenes_as_Pseudo_3D_Scenes_Using_2D_Diffusion.pdf]]
