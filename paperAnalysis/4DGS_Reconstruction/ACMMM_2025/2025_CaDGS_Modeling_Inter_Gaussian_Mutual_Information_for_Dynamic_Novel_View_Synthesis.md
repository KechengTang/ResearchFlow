---
title: "CaDGS: Modeling Inter-Gaussian Mutual Information for Dynamic Novel View Synthesis"
venue: ACMMM
year: 2025
tags:
  - 4DGS_Reconstruction
  - gaussian-splatting
  - 4dgs
  - mutual-information
  - correlation-aware
  - gctp
  - stdc
  - dynamic-novel-view-synthesis
  - specular-scene
  - status/analyzed
core_operator: 先用信息论视角显式建模 Gaussian 之间的互信息，再通过 GCTP 把原始 O(n^3) 相关张量压缩成 dual-channel O(n^2) 空间矩阵，结合 STDC 的多尺度一致性约束，让高斯形变从“独立预测”变成“相关联的协调变换”。
primary_logic: |
  输入动态场景后，CaDGS 不再把每个 Gaussian 的形变独立建模，
  而是先估计 inter-Gaussian mutual information，并通过 GCTP 投影为可卷积处理的双通道相关矩阵，
  再利用 STDC 在局部刚性与长程结构依赖上施加多尺度约束，
  最终让外观保持、运动传播与体渲染一致性在同一套相关性表示中被联合优化。
pdf_ref: paperPDFs/4DGS_Reconstruction/ACMMM_2025/2025_CaDGS_Modeling_Inter_Gaussian_Mutual_Information_for_Dynamic_Novel_View_Synthesis.pdf
category: 4DGS_Reconstruction
created: 2026-04-18T21:12
updated: 2026-04-18T21:12
---

# CaDGS: Modeling Inter-Gaussian Mutual Information for Dynamic Novel View Synthesis

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [DOI](https://doi.org/10.1145/3746027.3755121)
> - **Summary**: 这篇论文抓住了一个很核心但经常被忽略的问题: 现有 4DGS 大多把每个 Gaussian 的形变独立预测，但真实渲染质量其实来自多个高斯在相机射线上的协同作用。CaDGS 的关键贡献是把这种 “高斯之间的协同关系” 用 mutual information 显式建模出来，再把它压缩成可学习的相关矩阵去指导变形。
> - **Key Performance**:
>   - 在 Neu3D 上达到约 `32.38 PSNR / 0.967 SSIM / 0.038 LPIPS / 323 FPS / 190 MB`。
>   - 在 HyperNeRF 上达到 `26.12 PSNR / 0.89 MS-SSIM / 351 FPS / 190 MB`。
>   - 在 NeRF-DS 上平均达到 `24.74 PSNR / 0.9012 SSIM / 0.1351 LPIPS`，对高光和反射场景尤其有效。

---

## Part I / The "Skill" Signature

### 它到底在解决什么问题

CaDGS 解决的是：

**dynamic 4DGS 里每个 Gaussian 独立变形，导致整体体渲染结构不协调，从而引发几何扭曲、纹理不一致和视角相关伪影。**

现有很多方法默认：

- 每个 Gaussian 只看时间和自身特征
- deformation network 独立预测每个 Gaussian 的变化
- 只要单点预测得对，整体就会好

这篇论文明确反驳了这种隐含假设。作者通过 mutual information 分析认为：

- 局部邻近 Gaussian 之间存在 appearance-preserving radiance coherence
- 长距离语义相关 Gaussian 之间存在 motion-consistent deformation propagation

也就是说，**高斯不是独立粒子，而更像一个动态 radiance graph。**

### 它的方法直觉

方法直觉非常直接：

**如果渲染质量依赖于多个高斯协同变换，就应该把 inter-Gaussian dependency 显式编码进 deformation pipeline。**

因此 CaDGS 引入两个核心组件：

- `GCTP`: Gaussian Correlation Tensor Projection
- `STDC`: Spatio-Temporal Deformation Consistency

前者负责把复杂互信息结构投影成紧凑、可学习的相关矩阵；后者负责把这种相关性真正变成多尺度形变约束。

### 一句话能力画像

- **输入**：动态多视角视频场景
- **关键判断**：Gaussian deformation should be coordinated, not independent
- **核心 operator**：mutual-information projection + multi-scale consistency
- **目标**：提升 dynamic NVS 质量，同时保持实时渲染效率

### 对我当前方向的价值

这篇对你当前方向的价值很高，因为它补的是一个结构性缺口：

**很多 4DGS 工作只在问“怎么预测每个高斯”，CaDGS 在问“高斯之间应如何共同演化”。**

这对你现在的几条方向都重要：

- 对 `4DGS_Reconstruction`：它给出了一种显式建模相关性而不是只堆更强 MLP 的路线。
- 对 `4DGS_Editing`：如果后续要做结构保持型编辑，相关高斯的协同更新会比逐点独立编辑更稳。
- 对复杂材质和高光场景：它在 NeRF-DS 上的提升说明 correlation-aware operator 特别适合 specular / reflective dynamic scenes。

---

## Part II / High-Dimensional Insight

### 这篇论文真正新在哪

真正的新意主要有三层：

- **把 inter-Gaussian mutual information 作为一等建模对象**，而不是隐式期待网络自己学出来。
- **GCTP**：把原本 `O(n^3)` 的互信息张量压成 `O(n^2)` 的 dual-channel spatial matrix，同时保留外观一致性和运动一致性信号。
- **STDC**：通过 tensor-guided regularization 和多尺度空洞卷积，把局部刚性与长程结构传播统一到同一套约束里。

这不是简单加个正则项，而是把 4DGS 的 deformation 视角从 “independent points” 改成 “correlated field”。

### The "Aha!" Moment

最值得记住的 aha 是：

**动态场景里真正应该被建模的不是某个 Gaussian 的单点位移，而是它和其他 Gaussian 的相互依赖关系。**

这件事很重要，因为很多视觉伪影其实并不是某一个点错了，而是：

- 邻域里几个 Gaussian 没有一起动
- 远程语义相关区域没有同步保持结构
- appearance 与 motion 两套信号没有协同传播

CaDGS 试图把这三个问题一次性统一掉。

### 与 4DGS_Editing / 3DGS_Editing / feed-forward Gaussians 的关系

- **与 4DGS_Editing**：关系很近。编辑时如果只改局部 Gaussian，常会破坏结构一致性。CaDGS 提示后续可以在编辑传播中引入 correlation graph。
- **与 3DGS_Editing**：静态场景也存在局部编辑破坏整体连贯性的问题，GCTP 这种相关性压缩思路可以迁移到结构保持编辑。
- **与 feed-forward Gaussians**：这篇不是前馈模型，但它非常适合做 feed-forward Gaussian predictor 的辅助监督或蒸馏目标，因为它明确告诉模型“哪些高斯应该一起变”。

### 局限、风险、可迁移点

- **局限**：需要额外建模相关张量，系统复杂度比简单 deformation MLP 更高。
- **风险**：相关性建模若不准确，可能把错误的长程依赖也传递进去。
- **可迁移点**：
  - correlation-aware deformation instead of pointwise deformation
  - mutual-information compression for scalable structured learning
  - multi-scale local/global consistency in dynamic Gaussian fields

---

## Part III / Technical Deep Dive

### Pipeline

```text
dynamic scene
-> temporal Gaussian representation
-> estimate inter-Gaussian mutual information
-> GCTP projects O(n^3) tensor to dual-channel O(n^2) matrix
-> STDC applies multi-scale consistency constraints
-> coordinated deformation learning
-> dynamic novel view synthesis
```

### 关键模块

#### 1. Gaussian Correlation Tensor Projection

GCTP 是全文最核心的 operator。  
它把庞大的互信息张量压缩成一个双通道空间矩阵：

- 一个通道偏向 appearance-maintenance
- 一个通道偏向 motion-consistency

这样后续网络就可以用卷积去处理高斯相关结构，而不是直接在原始高阶张量上做学习。

#### 2. Spatio-Temporal Deformation Consistency

STDC 负责把相关性真正转化为可优化的几何约束。  
它一方面在局部维持 neighborhood rigidity，另一方面通过多尺度感受野维持全局结构传播，避免：

- 邻近高斯 jitter
- 长距离结构断裂
- 纹理与运动不同步

#### 3. Temporal Representation of Gaussian Properties

作者还特别做了 temporal representation 的消融，证明不能只对中心位置做时间编码。  
位置和非位置属性都要进 temporal network，才能捕获完整变形动态。

### 关键实验信号

- Neu3D 上对比 STG、4DGS、4DRotorGS 等方法时，CaDGS 同时在质量、速度、存储上都很强。
- HyperNeRF 上提升说明它不仅适合轻微动态，也能覆盖大非刚体形变。
- NeRF-DS 上的表现说明 cross-domain appearance/motion interaction 对高光和反射材质特别有帮助。
- Ablation 里去掉 GCTP、去掉 non-positional embedding、去掉 channel attention，PSNR 都明显下降，说明提升不是偶然。

### 对当前研究最可迁移的 operator

- **inter-Gaussian dependency modeling**
- **tensor-to-matrix correlation projection**
- **multi-scale deformation consistency for dynamic GS**

如果你之后要做复杂动态场景重建或结构保持型 4D 编辑，这篇值得作为 “高斯之间如何协同” 的基线参考。

## Local Reading / PDF 参考

![[paperPDFs/4DGS_Reconstruction/ACMMM_2025/2025_CaDGS_Modeling_Inter_Gaussian_Mutual_Information_for_Dynamic_Novel_View_Synthesis.pdf]]
