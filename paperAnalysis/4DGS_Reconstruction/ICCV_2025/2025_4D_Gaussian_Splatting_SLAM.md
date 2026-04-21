---
title: "4D Gaussian Splatting SLAM"
venue: ICCV
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - slam
  - rgb-d
  - online-mapping
  - optical-flow-supervision
  - status/analyzed
core_operator: 在 RGB-D 动态场景中联合执行相机跟踪与 4D Gaussian mapping，通过 motion masks 将高斯分成 static/dynamic sets，并用 sparse control points + MLP 建模动态变换场，再以重建出的 2D optical flow 监督动态高斯运动学习。
primary_logic: |
  先从 RGB-D 序列中生成 motion masks 估计静动态先验，
  再分别维护 static Gaussian set 与 dynamic Gaussian set，
  同时用 sparse control points 和 MLP 参数化 dynamic Gaussian 的 transformation field，
  最后结合 optical-flow reconstruction、photometric 与 geometric constraints 进行在线跟踪与 4D 场景建图。
pdf_ref: paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_4D_Gaussian_Splatting_SLAM.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T16:17
updated: 2026-04-18T16:17
---

# 4D Gaussian Splatting SLAM

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv 2503.16710](https://arxiv.org/abs/2503.16710)
> - **Summary**: 这篇论文的关键不是把 static GS-SLAM 扩成“能忽略动态物体的 SLAM”，而是尝试同时做 camera tracking 和 dynamic 4D mapping。也就是说，动态体不再被当成噪声删除，而是被纳入在线 4D Gaussian radiance field 的建图对象。
> - **Key Performance**:
>   - 论文在真实动态环境中报告更鲁棒的 tracking 和更高质量的 novel view synthesis。
>   - 文中的场景重建表明，相较 MonoGS/Gaussian-SLAM 等静态式基线，动态场景平均 PSNR/SSIM/LPIPS 有明显提升。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

传统 SLAM 在动态场景里通常采取一个保守策略:

- 检测动态体
- 尽量剔除动态像素
- 只重建静态环境

但这会直接放弃真实世界里很重要的动态结构信息。  
4D Gaussian Splatting SLAM 要解决的是:

**能不能在未知动态环境中，一边在线跟踪相机，一边把动态对象也纳入 4D 场景建图。**

### 核心能力定义

- **输入**: RGB-D sequence
- **输出**: camera poses + dynamic 4D Gaussian radiance fields
- **强项**: 在线 tracking、动态环境 mapping、动态 novel-view synthesis
- **弱项**: 纯单目设定、大规模开放世界长期运行、语义级控制

### 真正的挑战来源

- 动态体会干扰相机定位
- 如果全删动态像素，虽然 tracking 变稳，但 4D 重建直接缺失
- 在线系统里动态体的运动学习必须高效且增量可优化

### 边界条件

- 依赖 RGB-D 输入而非单纯 RGB
- 更强调 tracking + mapping 联合，而不是离线最优重建
- 需要 motion mask 与 optical flow 等辅助约束

---

## Part II / High-Dimensional Insight

### 方法的整体设计哲学

这篇论文最值得注意的是它的立场变化:

**动态对象不是 SLAM 里的 distractor，而是 4D mapping 里应该被建模的对象。**

因此系统从一开始就把静态和动态高斯拆成两个集合，各自承担不同职责。

### The "Aha!" Moment

真正的 aha 是:

**只要 tracking 仍由静态先验支撑，而动态高斯的运动又有单独监督，那么在线 SLAM 和 4D 动态建图并不冲突。**

作者用 motion masks 获得静动态 priors，用 sparse control points + MLP 学动态变换场，再通过重建 2D optical flow 来监督动态高斯，这构成了一个比较完整的在线闭环。

### 为什么这个设计有效

- static Gaussians 保证定位稳定性
- dynamic Gaussians 负责承接场景运动信息
- optical flow reconstruction 给动态体提供跨邻近帧的显式运动监督

因此系统既不会像传统 SLAM 那样简单扔掉动态体，也不会因为全场一起优化而让 tracking 崩掉。

### 对我当前方向的价值

这篇论文对你当前方向的价值很直接:

- 如果你后面关心实时 4DGS 或机器人场景表示，它是非常重要的 online 4D baseline
- 它说明 static/dynamic separation 不一定只是为重建服务，也可以服务 tracking 和交互

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**: 任何在线动态编辑系统都需要稳定追踪和场景增量更新，这篇提供的是“先把在线 4D map 建起来”的基础层。
- **与 3DGS_Editing**: 3DGS 编辑更多是离线对象，4DGS SLAM 则说明在动态环境里编辑前先要解决跟踪和增量建图。
- **与 feed-forward Gaussians**: 它不是 feed-forward，但它的 online split-and-update 思路很适合未来做 streaming Gaussian systems。

### 战略权衡

- 优点: tracking 与动态建图统一，真正面向在线场景
- 代价: 依赖 RGB-D、motion masks 和 flow supervision，系统复杂度高

---

## Part III / Technical Deep Dive

### Pipeline

```text
RGB-D sequence
-> motion masks for static/dynamic priors
-> static Gaussian set + dynamic Gaussian set
-> sparse control points and MLP for dynamic transformation
-> reconstructed 2D optical flow supervision
-> online pose tracking and 4D Gaussian mapping
```

### 关键模块

#### 1. Static / Dynamic Gaussian Sets

作者不是在同一套高斯里兼顾所有任务，而是显式区分 static 和 dynamic primitives。  
这有利于把 tracking 与 dynamic modeling 分工。

#### 2. Sparse Control Points For Dynamic Motion

动态高斯不是各自独立学习运动，而是由 sparse control points 与 MLP 共同定义 transformation field，兼顾表达力和在线效率。

#### 3. Optical Flow Reconstruction Supervision

这是很关键的一步。  
通过重建动态对象在邻近图像间的 2D optical flow，作者给动态高斯学习提供了更明确的运动监督。

### 关键实验信号

- 论文同时看 tracking robustness 和 view synthesis quality，指标设计符合系统目标
- 相比把动态体删掉的 SLAM 方案，它强调的是“在保稳的前提下保留动态信息”
- 真实场景实验是核心证据，因为这是在线系统最重要的落点

### 少量关键数字

- 文中多个动态场景上相对 MonoGS / Gaussian-SLAM / SplaTAM 等基线取得更好的平均重建指标

### 局限、风险、可迁移点

- **局限**: RGB-D 依赖较强，且在线系统很难达到离线重建质量
- **风险**: motion mask 或 flow reconstruction 误差会同时影响 tracking 与 mapping
- **可迁移点**: online static/dynamic split、flow-supervised dynamic Gaussians、incremental 4D mapping 都可迁移到实时 4D 应用

### 实现约束

- RGB-D 动态场景
- 在线 tracking + mapping 联合优化
- 强依赖 motion mask 与 optical flow supervision

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/ICCV_2025/2025_4D_Gaussian_Splatting_SLAM.pdf]]
