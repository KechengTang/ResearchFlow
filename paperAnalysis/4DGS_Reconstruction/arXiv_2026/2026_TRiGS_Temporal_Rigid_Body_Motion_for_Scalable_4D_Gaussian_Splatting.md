---
title: "TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting"
venue: arXiv
year: 2026
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - continuous-motion-modeling
  - rigid-body-motion
  - se3
  - bezier-residual
  - local-anchors
  - long-sequence
  - temporal-stability
  - status/analyzed
core_operator: 用连续的 SE(3) rigid-body motion、分层 Bézier residual 和 per-Gaussian local anchors 取代分段线性速度近似，把 4DGS 的 motion 建模从 temporally fragmented primitive resetting 改写成持续的几何一致变换。
primary_logic: |
  输入动态场景视频与 canonical Gaussians 后，先给每个 Gaussian 分配一个耦合的 SE(3) base motion，再用时间相关的 Bézier residual 平滑细化非线性轨迹，
  同时引入 local anchors 作为各自独立旋转中心保证局部 rigid motion 稳定，最后配合 motion-guided relocation 在固定 primitive budget 下循环回收低价值高斯，
  从而在长序列中保持 temporal identity 并抑制 Gaussian proliferation。
pdf_ref: paperPDFs/4DGS_Reconstruction/arXiv_2026/2026_TRiGS_Temporal_Rigid_Body_Motion_for_Scalable_4D_Gaussian_Splatting.pdf
category: 4DGS_Reconstruction
created: 2026-04-21T16:00
updated: 2026-04-21T16:00
---

# TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://wwwjjn.github.io/TRiGS-project_page/) | [arXiv 2604.00538](https://arxiv.org/abs/2604.00538)
> - **Summary**: TRiGS 把长视频 4DGS 里最核心的 failure mode 明确命名成 `temporal fragmentation`，并用连续 rigid-body motion 取代 piecewise linear velocity，把“高斯不断失活、重生、膨胀”的 makeshift 轨迹修补方式改写成持续、结构一致的运动表示。
> - **Key Performance**:
>   - 在 `SelfCap` 的 `1200` 帧设置下达到 `26.05 PSNR / 0.904 SSIM / 0.099 LPIPS`，优于 `FTGS` 的 `25.41 / 0.871 / 0.146`。
>   - 在统一 `500k` Gaussian budget 的 `1200` 帧对比中，仍取得平均 `26.05 PSNR / 0.904 SSIM / 0.099 LPIPS`，说明优势不是靠 primitive 无限制膨胀换来的。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？

TRiGS 瞄准的是当前显式 4DGS 在长时序里一个很本质的问题：

**很多高性能方法虽然在短片段上效果很好，但本质上还是用短 temporal window 或 piecewise linear velocity 去逼近连续非线性运动，结果就是高斯不断 drift、被迫重置、再 proliferate。**

作者把这件事概括成：

- temporal fragmentation
- long-term temporal identity 丢失
- memory footprint 随序列增长持续膨胀

### 核心能力定义

- **输入**：多视角动态视频。
- **输出**：带连续 rigid-body motion 的 4D Gaussian scene。
- **能做**：标准 4D novel-view synthesis，且能自然扩展到 `600 / 900 / 1200` 帧长序列。
- **不主打**：feed-forward 或跨场景泛化；它依然是 per-scene optimization。

### 真正的挑战来源

- 线性速度近似对长时域非线性轨迹误差会累计。
- 若 rotation 和 translation 独立优化，很容易互相抵消，导致 motion decomposition 不稳定。
- 一旦运动轨迹失真，系统通常通过 split / relocate / regenerate 高斯来补洞，结果就是 memory 爆炸。

### 边界条件

TRiGS 的 motion bias 明显更偏 rigid-body consistency。它通过 local anchors 给局部 rigid motion 更大自由度，但并不是专门为强非刚体拓扑变化设计的。

## Part II / High-Dimensional Insight

### 方法真正新的地方

TRiGS 的真正新意不只是“把 motion 写得更复杂”，而是提出一个判断：

**长时序 4DGS 的主问题不是 primitive 不够灵活，而是它们太碎、太短命，无法维持长期 temporal identity。**

因此它选择反方向：

- 少一点 fragmented flexibility
- 多一点 geometrically coupled continuous motion

### The "Aha!" Moment

真正的 aha 是：

**如果 Gaussian primitive 的运动不是一段段线性拼接，而是被一个连续的 SE(3) 变换和时间平滑 residual 共同驱动，那么“高斯为什么一直要死掉重生”这个问题会被直接削弱。**

这条链路的因果关系非常清楚：

- 用 SE(3) 把 rotation 和 translation 耦合起来
- 用 quadratic Bézier residual 补充非线性时间变化
- 用 per-Gaussian local anchor 让每个 primitive 有自己的局部旋转中心
- 于是 motion 更连续，track identity 更久，高斯不需要频繁 split/reinit 去掩盖轨迹误差

### 为什么这个设计有效？

- SE(3) 耦合约束避免了 rotation / translation 两套参数在 photometric loss 下相互补偿、越学越飘。
- Bézier residual 给它补足了非线性动态，不至于 rigid formulation 过硬。
- local anchors 让“局部 rigid motion”成立，而不是强迫所有 primitive 围着统一 global center 转。

### 战略权衡

- **优点**：长序列更稳，memory 更受控，temporal identity 更连贯。
- **代价**：表征偏强结构化；如果任务需要特别剧烈的非刚体/拓扑变化，未必是最自由的选择。

## Part III / Technical Deep Dive

### Pipeline

```text
canonical Gaussians
-> parameterize each primitive with coupled SE(3) base motion
-> add hierarchical quadratic Bézier residual over time
-> attach learnable local anchors as independent rotation centers
-> deform means/covariances continuously over time
-> optimize with rendering loss + motion smoothness + local rigid coherence
-> periodically relocate low-value primitives under fixed budget
```

### 关键机制

#### 1. Coupled rigid-body motion

TRiGS 不再让 primitive 的 translation 和 rotation 独立地各走各的，而是直接在 `se(3)` 上参数化，再通过 exponential map 映射到 `SE(3)`。这一点很关键，因为它把 motion 从“多个会互相抵消的自由参数”变成“一个几何上更一致的整体变换”。

#### 2. Hierarchical Bézier residual

光有 base rigid transform 不够表达复杂 motion，所以它又加了 quadratic Bézier residual。这个 residual 不是为了推翻 base motion，而是为了在连续时间上平滑修正轨迹弯曲程度。

#### 3. Local anchors

局部旋转中心不是全局共享，而是每个 Gaussian 都有自己的 learnable local anchor。这样更像在做局部 part-wise rigid motion，而不是整个场景围着一个轴转。

#### 4. Motion-guided relocation

TRiGS 没有放弃 relocation，但它把 relocation 从“不得不疯狂补洞”的救火策略，改成“在固定 primitive budget 下回收低价值高斯”的 capacity management 策略。

### 关键实验信号

- 在 `SelfCap` 的 `600 / 900 / 1200` 帧设置下，TRiGS 平均都优于 4DGS / STGS / LocalDyGS / FTGS，说明它不是只在极长序列上勉强保住，而是在正常 benchmark 上也有竞争力。
- 最重要的信号是：当序列从 `600` 帧变到 `1200` 帧，很多 baseline 分数掉得很明显，而 TRiGS 下降更少，说明它确实在解决 long-range temporal stability。
- 在 matched `500k` budget 对比里它仍然最强，说明优势不是因为容忍更多 Gaussians，而是 motion formulation 本身更合理。

### 对你最有价值的启发

对你做 `4DGS editing`，TRiGS 最值钱的地方不是直接拿来当 baseline，而是给你一个非常清楚的 failure framing：

- 长时序里真正会坏掉的是 temporal identity
- identity 一坏，edit persistence 也会坏
- 如果你的 A0 后面出现 long-range edit drift，TRiGS 这种 continuous temporal carrier 会是很自然的升级方向

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/arXiv_2026/2026_TRiGS_Temporal_Rigid_Body_Motion_for_Scalable_4D_Gaussian_Splatting.pdf]]
