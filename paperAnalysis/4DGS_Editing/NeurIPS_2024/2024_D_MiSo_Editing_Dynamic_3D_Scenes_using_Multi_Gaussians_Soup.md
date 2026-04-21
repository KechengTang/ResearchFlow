---
title: "D-MiSo: Editing Dynamic 3D Scenes using Multi-Gaussians Soup"
venue: NeurIPS
year: 2024
tags:
  - 4DGS_Editing
  - gaussian-splatting
  - 4dgs
  - dynamic-scene-editing
  - mesh-hybrid
  - scene-manipulation
  - motion-editing
  - status/analyzed
core_operator: 用 mesh-inspired multi-gaussian triangle soup 重组动态高斯场景，把对象运动与渲染高斯解耦成可编辑的结构单元，从而支持时序上的动态操控。
primary_logic: |
  输入动态场景序列与相机位姿，先估计 mesh/triangle soup 结构并把高斯链接到这些可参数化单元上，
  再用“大高斯做全局变换、小高斯做局部渲染”的 multi-Gaussian 设计和双 deformation networks 建模时序运动，
  最后通过修改轨迹而不是逐帧重做编辑，实现可持续传播的动态场景编辑。
pdf_ref: paperPDFs/4DGS_Editing/NeurIPS_2024/2024_D_MiSo_Editing_Dynamic_3D_Scenes_using_Multi_Gaussians_Soup.pdf
category: 4DGS_Editing
created: 2026-04-18T14:30
updated: 2026-04-18T14:30
---

# D-MiSo: Editing Dynamic 3D Scenes using Multi-Gaussians Soup

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [arXiv](https://arxiv.org/abs/2405.14276) / [PDF](https://arxiv.org/pdf/2405.14276.pdf) / [GitHub](https://github.com/waczjoan/D-MiSo)
> - **Summary**: D-MiSo 不是走文本驱动路线，而是把动态高斯场景改造成更像 mesh 的可编辑结构，让“改动态”从难以复现的逐帧/逐点干预，变成对时序轨迹和结构单元的直接操控。
> - **Key Performance**:
>   - 论文报告在 D-NeRF 数据上保持了与主流动态重建方法相近的重建质量。
>   - 同时它额外提供了时序可编辑性，这是多数纯重建型 4DGS 方法不具备的。

## Part I / The "Skill" Signature

### 它到底在解决什么问题

这篇论文解决的是动态 3D 场景编辑里最现实的问题之一：

**很多动态 GS 方法能表示和渲染运动，但一旦你想把一个对象“沿时间改掉”，编辑接口就非常脆弱。**

此前像 SC-GS 一类方法已经在尝试做可编辑动态场景，但仍然比较依赖人工指定：

- 哪些部分保持固定
- 哪些控制点要跟着动
- 不同时间上的改动如何可复现地传播

D-MiSo 的目标就是把这件事系统化。

它的能力边界也很清楚：

- 强项：对象级、轨迹级、结构级的动态编辑
- 弱项：纯文本语义编辑、极复杂非刚体拓扑变化

## Part II / High-Dimensional Insight

### 方法真正新在哪里

D-MiSo 的关键是把动态高斯场景组织成一种 **mesh-inspired triangle soup**。

这不是为了回到传统 mesh，而是为了获得一个更稳定的编辑界面：

- mesh/triangle 提供结构约束
- linked Gaussians 提供渲染表现力
- 轨迹参数化负责时间传播

### The "Aha!" Moment

真正的 aha 在于：

**动态场景难编辑，不只是因为运动复杂，而是因为缺少一个“能承载编辑”的结构骨架。**

D-MiSo 给出的骨架是：

1. 用 triangle soup 组织动态对象
2. 用 multi-Gaussian 组件把全局运动和局部外观分开
3. 用 deformation networks 把运动写成可控轨迹

这样编辑不再是“某一帧改了，后面尽量跟上”，而是“先改运动结构，再让渲染跟随它”。

对你的 4DGS 编辑路线，这个思路很重要，因为它直接对应一个核心问题：
**你到底是在编辑视觉结果，还是在编辑驱动动态的结构变量？**

### 权衡与局限

- 优势：可复现、可传播、适合对象级动态编辑
- 局限：对结构估计质量敏感，不是语言驱动编辑范式

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene observations
-> estimated mesh / triangle soup
-> linked multi-Gaussian components
-> dual deformation modeling over time
-> edited trajectories for scene objects
-> editable dynamic Gaussian scene
```

### 关键技术点

#### 1. Multi-Gaussian Components

论文把单个对象的表示分成不同职责：

- 大高斯负责全局变换
- 小高斯负责细节渲染

这种职责分离让运动编辑不必直接冲击所有渲染细节。

#### 2. Triangle Soup as Editing Interface

作者真正提供的是一个更适合编辑的中间层。

相比直接拖动大量高斯，triangle soup 更接近可操控的结构对象，因此更容易定义长期一致的轨迹修改。

#### 3. Dynamic Editing over Time

论文强调的不是“单帧好改”，而是“修改后能沿时间持续传播”。

这也是它和很多只会动态重建的方法最本质的差别。

### 关键信号

- 论文在 D-NeRF 风格数据上给出了与其他动态方法可比的重建质量。
- 更关键的是，它把动态编辑接口显式化了，而不是把编辑当成重建后的附赠品。

### 对你的价值

如果你之后要做 4DGS 编辑，这篇论文最值得保留在核心阅读集里的原因是：

- 它提供了 **structure-first** 的动态编辑思路
- 它和 Instruct-4DGS / CTRL-D 这类图像编辑驱动路线形成明显互补

前者更像在回答“改什么结构量”，后者更像在回答“如何把编辑意图注入视觉生成器”。

## Local Reading / PDF 引用

![[paperPDFs/4DGS_Editing/NeurIPS_2024/2024_D_MiSo_Editing_Dynamic_3D_Scenes_using_Multi_Gaussians_Soup.pdf]]
