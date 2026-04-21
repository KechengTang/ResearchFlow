---
title: "Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting"
venue: NeurIPS
year: 2025
tags:
  - Gaussian_Splatting_Foundation
  - gaussian-splatting
  - 4dgs
  - compression
  - rate-distortion-optimization
  - wavelet-trajectory
  - dynamic-scene-compression
  - status/analyzed
core_operator: 在 Ex4DGS 的静态/动态高斯分解上叠加 RD-aware pruning、entropy-constrained vector quantization 与轨迹级 Haar wavelet 压缩，把 4DGS 从“能渲染”推进到“可按码率部署”的端到端压缩表示。
primary_logic: |
  先训练 Ex4DGS 得到静态和动态高斯，
  再对共享属性做 masking 与 entropy-constrained vector quantization，
  同时把动态点位轨迹做 pointwise Haar wavelet 变换并裁掉高频细节，
  最后用联合 rate-distortion loss 在码率、画质与时序平滑之间做端到端折中。
pdf_ref: paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2025/2025_Temporal_Smoothness_Aware_Rate_Distortion_Optimized_4D_Gaussian_Splatting.pdf
category: Gaussian_Splatting_Foundation
created: 2026-04-18T16:49
updated: 2026-04-18T16:49
---

# Temporal Smoothness-Aware Rate-Distortion Optimized 4D Gaussian Splatting

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2507.17336](https://arxiv.org/abs/2507.17336) | [Code](https://github.com/HyeongminLEE/RD4DGS)
> - **Summary**: 这篇论文关注的不是再发明一个新的 4DGS 表示，而是把已有的 Ex4DGS 改造成真正可部署的压缩表示。核心抓手是把 3DGS 里已有的 RD-aware 量化思路扩展到动态场景，并显式利用“动态点轨迹大多是平滑的”这一时序先验。
> - **Key Performance**:
>   - 在 N3V 上，`L1` 压缩级别把 Ex4DGS 的平均模型大小从 `115 MB` 压到 `1.26 MB`，约 `98.9%` 缩减，同时渲染速度达到 `163 FPS`。
>   - 在 Technicolor 上，`L6` 仍可把模型压到 `19.6 MB` 且保持 `32.20 PSNR / 113.1 FPS`；全文报告最高约 `91x` 压缩。

## Part I / The "Skill" Signature

### 它到底在解决什么问题？
4DGS 这条线的一个现实瓶颈不是“能不能重建”，而是“能不能存、能不能传、能不能部署”。  
显式高斯表示虽然已经比很多隐式方法快，但一旦进入动态场景，模型要同时存：

- 静态高斯属性
- 动态高斯属性
- 关键帧位移或时序轨迹
- 时序相关的透明度和旋转参数

结果就是模型尺寸迅速膨胀。论文要解决的是：

**如何把 4DGS 从研究型表示变成一个可按 bit budget 调节的、可在不同算力平台上部署的压缩表示。**

### 核心能力定义

- **输入**: 已训练或待微调的 Ex4DGS 动态高斯表示
- **输出**: 经过 RD 优化的压缩 4DGS bitstream / 参数集合
- **强项**: 码率-画质可控、显式模型可压缩、保持实时渲染
- **弱项**: 重点是压缩与部署，不负责提升底层 4D 重建上限

### 真正的挑战来源

- 4DGS 比 3DGS 多了一个时间轴，冗余不再只是空间冗余
- 动态点轨迹如果逐关键帧独立存储，会浪费大量时序 redundancy
- 不是所有属性都适合同样的量化强度，错误量化会明显拉低保真度
- 4DGS 压缩不能只追求“更小”，还必须兼顾渲染质量与速度

### 边界条件

- 方法建立在 Ex4DGS 的 static/dynamic decomposition 之上
- 更适合轨迹相对平滑、可被低频运动描述的动态场景
- 目标是场景整体压缩，不是逐帧视频流式编码器

## Part II / High-Dimensional Insight

### 方法的整体设计哲学
这篇论文最重要的判断是：

**4DGS 的压缩难点不是简单把 3DGS 压缩模块照搬过去，而是要专门处理“动态轨迹”这一新增信息通道。**

所以作者没有把动态位移当成普通参数无差别量化，而是拆成两层：

- 共享 3DGS 属性继续走 pruning + ECVQ 的经典路径
- 动态轨迹单独走 wavelet-based temporal compression

这让“空间属性压缩”和“时间轨迹压缩”被明确解耦。

### The "Aha!" Moment

真正的 aha 在于：

**动态 4D Gaussian 的时间变化不是离散孤立点，而是一条通常较平滑的轨迹；既然它是轨迹，就应该按信号处理的方式压，而不是按独立参数硬编码。**

作者因此把每个动态点的时间轨迹做单层 Haar wavelet 变换，只保留主要低频分量、削弱高频细节。这个操作相当于把“动态高斯的位移表”改写成“低频运动骨架 + 少量残差”。

### 为什么这个设计有效？

- 轨迹平滑先验与真实动态场景相匹配，尤其对连续运动物体有效
- wavelet 压缩直接打在最贵的动态位移通道上，比只压 SH 或透明度更击中要害
- RD loss 让不同属性的压缩强度不靠手工拍脑袋，而由码率和失真共同决定
- selective quantization 避免把对画质特别敏感的参数一刀切压坏

### 对我当前方向的价值
对你现在的 Gaussian Splatting 方向，这篇论文的价值非常直接：

- 如果以后做 `4DGS_Reconstruction`，它给的是动态高斯结果的部署层与存储层方案
- 如果以后做 `4DGS_Editing`，它提醒一个关键问题：编辑后的动态高斯如何被高效保存、传输和回放
- 如果你关注 feed-forward Gaussians，这篇论文提供的是一个后处理视角：预测出高斯只是第一步，真正落地还要面对 entropy、bit allocation 与 streaming 代价

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 关系最强。编辑系统如果最终产出动态 Gaussian 序列，就会立刻遇到模型体积和时序属性存储问题；轨迹级 wavelet 压缩很适合作为 editing 后端。
- **与 3DGS_Editing**: 关系是“静态压缩如何升级到动态压缩”的延伸。它说明 editing 不该只考虑外观改动，还要考虑修改后表示的码率代价。
- **与 feed-forward Gaussians**: 关系在部署层而不在训练层。前馈模型往往把生成速度做快，但模型落地仍受模型大小与传输约束；RD-aware 压缩可以接在预测结果之后。

### 战略权衡

- 优点: 真正把 4DGS 推向“可部署表示”，而不只是学术 benchmark
- 代价: 画质上限仍受 Ex4DGS 底座影响，且强压缩时动态区域更容易先受损

## Part III / Technical Deep Dive

### Pipeline

```text
pretrained Ex4DGS
-> split static / dynamic Gaussian attributes
-> Gaussian pruning + SH pruning + entropy-constrained vector quantization
-> pointwise Haar wavelet transform on dynamic trajectories
-> selective quantization on opacity-related terms
-> end-to-end RD optimization
-> compressed 4DGS with controllable bitrate-quality tradeoff
```

### 关键模块

#### 1. RD-Aware Masking And ECVQ
作者延续 RD3DGS 的思路，对高斯数量和 SH 系数做可学习 pruning，并配合 entropy-constrained vector quantization。  
这部分负责压掉共享显式表示里的冗余。

#### 2. Dynamic Trajectory Wavelet Compression
真正的新东西在这里。  
对于动态高斯的 $\mu_d(t)$，作者沿时间轴做单层 Haar wavelet 分解，只保留主要低频系数并去掉高频细节，从而利用运动平滑性。

#### 3. Selective Quantization On Opacity Terms
论文专门分析了动态 opacity 的不同分量。结论不是“全量化越狠越好”，而是要对不同透明度参数做选择性量化，否则码率收益有限但画质损失明显。

#### 4. End-to-End RD Objective
最终训练目标把失真项、码率项和 Ex4DGS 自身的正则项绑在一起，使压缩器与场景表示共同优化，而不是先训练好场景再做纯离线编码。

### 关键实验信号

- wavelet 轨迹压缩的 RD 曲线始终优于不做 wavelet 的版本，说明“轨迹先验”不是装饰，而是真带来 size-quality 优势
- 在 N3V 上，`L1` 平均 `1.26 MB / 163 FPS / 27.04 PSNR`，`L6` 平均 `11.06 MB / 100.9 FPS / 29.66 PSNR`
- 在 Technicolor 上，`L1` 可压到 `2.1 MB / 213.9 FPS`，`L6` 为 `19.6 MB / 113.1 FPS`
- 与并行工作 Light4GS 对比时，它在高压缩设定下更小更快，表明其目标函数确实偏向“可部署 trade-off”而不是峰值 PSNR

### 少量关键数字

- 最高约 `91x` 压缩
- N3V 平均尺寸从 `115 MB` 压到 `1.26 MB` 或 `11.06 MB`
- 高压缩设定下约 `163 FPS`

### 局限、风险、可迁移点

- **局限**: 快速、剧烈、局部高频运动会被 wavelet 压缩得更模糊，文中明确提到高压缩下会出现 motion blur
- **风险**: 动态点在高保真设定下仍占大量 bit budget，继续粗暴剪枝会先伤害前景显著区域
- **可迁移点**: trajectory-as-signal、static/dynamic 分离码率预算、属性级 selective quantization 这三点都很值得迁移到其他 4DGS 与 editable Gaussian 系统

### 实现约束

- 训练流程是两阶段：先按 Ex4DGS 预训练，再做 RD-aware fine-tuning
- 主要实验在 RTX 3090 上完成，总训练时间约 2 小时
- 当前框架默认基于 Ex4DGS，但论文也明确表示核心压缩思想可迁移到其他 4DGS 底座

## Local Reading / PDF 参考
![[paperPDFs/Gaussian_Splatting_Foundation/NeurIPS_2025/2025_Temporal_Smoothness_Aware_Rate_Distortion_Optimized_4D_Gaussian_Splatting.pdf]]
