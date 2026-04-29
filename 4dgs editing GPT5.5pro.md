下面按“直接 4D/4DGS 编辑论文”和“进入该方向必须纳入视野的实时、动作、前馈式 4D Gaussian 论文”分开。严格筛选口径是：2024 年以来；已在 AI/视觉/图形顶会顶刊或其正式 proceedings / OpenReview 页面出现；arXiv-only 不进入主表。需要说明一点：Dynamic-eDiTor 和 Catalyst4D 的项目页均标注 CVPR 2026，但我没有把它们和已经有 CVF proceedings 页面的 CVPR 2024/2025 论文等量看待，而是标为“CVPR 2026 标注，待 proceedings 最终核验”。

开源完整度的判定标准如下：
“完整可复现”表示有训练、推理、评测、数据准备或预训练权重路径，按 README 基本能跑；“基本可复现”表示核心代码可用，但数据、checkpoint、评测脚本或若干配置需要人工补齐；“部分开源”表示只有关键脚本或 demo，难以完整复现实验；“未开源/空壳”表示代码 coming soon、无实质代码，或只有项目页。

## 1. 直接相关：4D / 4DGS 编辑主线

| 论文                                                         | Venue                                         | Base model / 表示                                            | Dataset                                                      | 对比 baseline                                                | 指标与实验                                                   | 开源情况                                                     |
| ------------------------------------------------------------ | --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Control4D: Efficient 4D Portrait Editing with Text**       | CVPR 2024                                     | 面向动态人像的 4D 表示；论文使用 **GaussianPlanes** 表示 3D+时间，并从不一致的 2D diffusion edits 中学习 4D generator / 4D portrait prior。项目页强调目标是文本驱动的 4D portrait editing，而不是通用动态场景动作编辑。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Shao_Control4D_Efficient_4D_Portrait_Editing_with_Text_CVPR_2024_paper.html)) | 主要是动态人像/portrait 序列；不是 DyNeRF/HyperNeRF 这类通用动态场景 benchmark 的主线方法。 | 主要对比 portrait / diffusion editing 类方法；后续 CTRL-D 明确提到 Control4D 代码未释放，因此很难作为标准可复现 baseline。([ar5iv](https://ar5iv.org/abs/2412.01792)) | 以文本编辑质量、身份保持、时序一致性、训练/编辑效率为核心；更适合“人像外观/局部属性编辑”，不适合作为通用动作编辑的核心 baseline。 | **未完整开源**。项目页显示 code “coming soon”；按当前可用信息，应视为不可复现实验。([Control4D](https://control4darxiv.github.io/)) |
| **Sparse-Controlled Gaussian Splatting for Editable Dynamic Scenes, SC-GS** | CVPR 2024                                     | **3DGS + 稀疏控制点 + 形变网络**。把动态场景分解为少量可控点和大量 3D Gaussian；每个 Gaussian 由控制点驱动 6-DoF 变换，并用 ARAP 约束保持局部刚性。该设计天然支持 motion / geometry editing。([arXiv](https://arxiv.org/abs/2312.14937)) | **D-NeRF、NeRF-DS**；编辑展示包括控制点拖拽、局部运动控制等。([ar5iv](https://ar5iv.org/html/2312.14937v3)) | D-NeRF、TiNeuVox-B、Tensor4D、K-Planes、FF-NVS、4D-GS、HyperNeRF、NeRF-DS 等。([ar5iv](https://ar5iv.org/html/2312.14937v3)) | 标准重建指标：PSNR、SSIM / MS-SSIM、LPIPS。补充实验表中 D-NeRF 平均约 **43.31 PSNR / 0.997 SSIM / 0.0063 LPIPS**；NeRF-DS 平均约 **24.1 PSNR / 0.891 MS-SSIM / 0.140 LPIPS**。编辑实验以可视化和交互控制为主，缺少统一动作编辑指标。([ar5iv](https://ar5iv.org/html/2312.14937v3)) | **基本完整可复现**。官方 GitHub 为 MIT license，包含 dynamic NeRF / Gaussian splatting / motion editing 主题代码。([GitHub](https://github.com/CVMI-Lab/SC-GS?utm_source=chatgpt.com)) |
| **Instruct 4D-to-4D: Editing 4D Scenes as Pseudo-3D Scenes Using 2D Diffusion** | CVPR 2024                                     | 不是 4DGS，而是 **NeRFPlayer / pseudo-3D dynamic scene editing**。使用 anchor-aware InstructPix2Pix、深度投影和光流外观传播，解决多视角/多时刻 2D diffusion edits 的不一致问题。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Mou_Instruct_4D-to-4D_Editing_4D_Scenes_as_Pseudo-3D_Scenes_Using_2D_CVPR_2024_paper.html)) | **DyCheck、HyperNeRF、DyNeRF / N3DV**。([ar5iv](https://ar5iv.org/html/2406.09402v1)) | IN2N-4D、FateZero / video editing 变体、若干 ablation。([ar5iv](https://ar5iv.org/html/2406.09402v1)) | PSNR、SSIM、LPIPS、CLIP 相关指标、运行时间。coffee_martini 等场景上相较 IN2N-4D 有更高重建/编辑一致性；论文也做 anchor、optical flow 等 ablation。([ar5iv](https://ar5iv.org/html/2406.09402v1)) | **基本可复现**。官方 GitHub 公开，MIT license。([GitHub](https://github.com/Friedrich-M/Instruct-4D-to-4D)) |
| **D-MiSo: Editing Dynamic 3D Scenes using Multi-Gaussians Soup** | NeurIPS 2024                                  | **Deformable Gaussian + mesh-inspired Multi-Gaussians Soup**。使用 Triangle Soup、Core-Gaussians、Sub-Gaussians，把运动和外观组织成更适合编辑的 mesh-like Gaussian 结构；论文明确指出 SC-GS 依赖固定/静态控制点，D-MiSo 试图解决其局限。([OpenReview](https://openreview.net/forum?id=3og0FT85B2&referrer=[the+profile+of+Przemysław+Spurek](%2Fprofile%3Fid%3D~Przemysław_Spurek1))) | **D-NeRF 7 个合成动态物体、NeRF-DS 7 个真实场景、PanopticSports 6 个动态运动场景**。([arXiv](https://arxiv.org/html/2405.14276v3)) | D-NeRF、TiNeuVox-B、Tensor4D、K-Planes、FF-NVS、4D-GS、DynMF、SC-GS、HyperNeRF、NeRF-DS 等。([arXiv](https://arxiv.org/html/2405.14276v3)) | PSNR、SSIM、LPIPS。D-NeRF 平均约 **42.27 / 0.993 / 0.0065**；NeRF-DS 平均约 **24.0 / 0.849 / 0.198**。动作/形变编辑仍主要是 qualitative，缺少标准动作幅度与动作准确性 benchmark。([arXiv](https://arxiv.org/html/2405.14276v3)) | **基本可复现**。官方代码公开。([GitHub](https://github.com/waczjoan/D-MiSo)) |
| **Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation, Instruct-4DGS** | CVPR 2025                                     | **4DGS canonical 3D Gaussians + HexPlane deformation**。核心做法是把动态 4DGS 拆成静态 canonical Gaussians 和 deformation field，在 canonical 空间编辑外观，再用固定 deformation 传播到全时序。论文明确把目标定义为 appearance editing，而非 motion editing。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Kwon_Efficient_Dynamic_Scene_Editing_via_4D_Gaussian-based_Static-Dynamic_Separation_CVPR_2025_paper.html)) | **DyNeRF**，6 个 10 秒动态场景，15–20 个 cameras，30 fps。([ar5iv](https://ar5iv.org/html/2502.02091v3)) | Instruct 4D-to-4D 是主要 baseline。([ar5iv](https://ar5iv.org/html/2502.02091v3)) | PSNR、SSIM、LPIPS、CLIP、运行时间。报告平均约 **27.22 PSNR / 0.856 SSIM / 0.250 LPIPS / 0.256 CLIP**，约 **40 min / 1 A40 GPU**；Instruct 4D-to-4D 约 **20.40 / 0.736 / 0.491 / 0.230**，约 **2 h / 2 GPUs**。([ar5iv](https://ar5iv.org/html/2502.02091v3)) | **基本可复现**。官方 GitHub 公开。([GitHub](https://github.com/juhyeon-kwon/efficient_4d_gaussian_editing)) 但论文 limitation 明确指出它不适合 motion editing，且局部编辑依赖 segmentation。([ar5iv](https://ar5iv.org/html/2502.02091v3)) |
| **CTRL-D: Controllable Dynamic 3D Scene Editing with Personalized 2D Diffusion** | CVPR 2025                                     | **Deformable 3DGS / 4DGS + personalized InstructPix2Pix**。用户先编辑单张参考图，方法用该参考图个性化 2D diffusion，再进行两阶段 4D scene optimization。适合局部外观/对象属性编辑。([Kai He](https://ihe-kaii.github.io/CTRL-D/)) | **DyCheck、N3DV，以及作者采集的人体中心动态场景**。([ar5iv](https://ar5iv.org/abs/2412.01792)) | Instruct 4D-to-4D；论文说明 Control4D 代码未释放，因此未作为可复现实验 baseline。([ar5iv](https://ar5iv.org/abs/2412.01792)) | CLIP Score、VBench / DINO subject consistency、运行时间。报告的场景如 portrait、cat、steak 中，CTRL-D 通常约 40–60 min 级别，而 Instruct 4D-to-4D 约 2 h 级别。([ar5iv](https://ar5iv.org/abs/2412.01792)) | **基本可复现**。官方实现公开，README 说明其基于 Deformable-3D-Gaussians、4DGaussians、TIP-Editor 等代码。([GitHub](https://github.com/IHe-KaiI/CTRL-D)) |
| **Dynamic-eDiTor: Direct 4D Editing of Dynamic Scenes using 3D Gaussian Splatting** | 项目页标注 CVPR 2026；需最终 proceedings 核验 | **预训练 4DGS + 多模态 DiT / Qwen-Image-Edit + 4D optimization**。提出 Spatial-Temporal Gaussian Alignment, STGA 和 Consistent Temporal Propagation, CTP，目标是直接优化源 4DGS，不重新训练独立 4D 表示。([Di Lee](https://di-lee.github.io/dynamic-eDiTor/)) | **DyNeRF、HyperNeRF**；实验详细设置包括 DyNeRF 6 个场景、16–21 views、10 秒 30 fps，采样到 1 fps，14 个编辑 prompt。([arXiv](https://arxiv.org/html/2512.00677v1)) | Instruct 4D-to-4D、Instruct-4DGS、CTRL-D。([arXiv](https://arxiv.org/html/2512.00677v1)) | CLIP directional similarity、CLIP similarity、PSNR、SSIM、LPIPS、MEt3R 多视角一致性、RAFT warping temporal consistency、user study、运行时间。作者报告 coffee_martini 全流程约 51 min / H100。([arXiv](https://arxiv.org/html/2512.00677v1)) | **基本可复现但仍偏早期**。GitHub 已公开，包含 script、src、requirements、pretrained scene / dataset 说明和 editing commands；但 no releases，仍需手动配置。([GitHub](https://github.com/DI-LEE/dynamic_eDiTor)) |
| **Catalyst4D: A 3D-Centric Framework for 4D Scene Editing**  | 项目页标注 CVPR 2026；需最终 proceedings 核验 | **3D-to-4D editing framework**。先在第一帧/3D Gaussian 上用强 3D editor 完成编辑，再通过 Adaptive Multi-view Gaussian Alignment, AMG 和 Consistent Uncertainty-Aware Refinement, CUAR 传播到 4D。多相机动态场景基于 Swift4D，单目动态场景基于 4DGS。([junliao2025.github.io](https://junliao2025.github.io/Catalyst4D-ProjectPage/)) | **DyNeRF、MeetRoom、HyperNeRF**。([ar5iv](https://ar5iv.org/html/2603.12766v2)) | 4D 编辑 baseline：Instruct 4D-to-4D、CTRL-D、Instruct-4DGS；3D 编辑模块还涉及 DGE、DreamCatalyst、SGSST 等。([ar5iv](https://ar5iv.org/html/2603.12766v2)) | CLIP similarity、temporal consistency score、运行时间。表中如 sear-steak 场景 Catalyst4D 约 **0.252 CLIP / 0.983 consistency / 50 min**；coffee-martini 约 **0.249 / 0.986 / 50 min**；trimming 约 **0.251 / 0.967 / 40 min**。([ar5iv](https://ar5iv.org/html/2603.12766v2)) | **未完整开源**。项目页显示 code coming soon。([junliao2025.github.io](https://junliao2025.github.io/Catalyst4D-ProjectPage/)) |

结论很直接：如果你的目标是“动作编辑准确性、连贯性、可实现动作变化幅度”，不能把 Instruct-4DGS / CTRL-D / Catalyst4D 这条外观编辑路线当作充分 baseline。Instruct-4DGS 自己的 limitation 已明确说明不适合 motion editing。([ar5iv](https://ar5iv.org/html/2502.02091v3)) 真正和动作/几何运动编辑强相关的顶会论文目前主要是 **SC-GS** 和 **D-MiSo**，再加上下面这些动作表示、前馈 4DGS、实时 4DGS 论文作为支撑 baseline。

## 2. 必须关注的实时、动作表示、前馈式 4D Gaussian 相关论文

| 论文                                                         | Venue                  | 为什么相关                                                   | Base model / Dataset / Baseline / 指标                       | 开源情况                                                     |
| ------------------------------------------------------------ | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **4D Gaussian Splatting for Real-Time Dynamic Scene Rendering** | CVPR 2024              | 4DGS 编辑论文最常用底座之一；Instruct-4DGS、Dynamic-eDiTor 等都围绕 canonical 3DGS + deformation / 4DGS 表示展开。 | 目标是高分辨率动态场景的快速训练与实时渲染；项目页称可在约 30 分钟内训练，并在 RTX3090 上 30+ fps，CVPR 页面报告 800×800 下 82 fps。([Guanjun Wu](https://guanjunwu.github.io/4dgs/)) | **完整可复现**。官方 GitHub 提供 data preparation、training、checkpoint、rendering 等。([GitHub](https://github.com/hustvl/4DGaussians)) |
| **Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting** | ICLR 2024              | 另一条实时 4DGS 基线；如果你强调实时编辑，至少要区分“实时渲染”和“实时编辑”。 | OpenReview 正式 ICLR 2024；使用 4D Gaussian primitives 与 4D spherindrical harmonics 建模动态场景。([OpenReview](https://openreview.net/forum?id=WhgB5sispV)) | **基本可复现**。官方代码公开。([GitHub](https://github.com/fudan-zvg/4d-gaussian-splatting)) |
| **Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis** | CVPR 2024              | 实时动态 view synthesis 的强 baseline；其时空 Gaussian 和 feature splatting 对后续语义/编辑控制有参考价值。 | 使用 temporal opacity、参数化 motion / rotation，报告 8K lite 60 fps 级别结果。([arXiv](https://arxiv.org/abs/2312.16812)) | **基本可复现**。官方 CVPR 2024 GitHub 公开。([GitHub](https://github.com/oppo-us-research/SpacetimeGaussians)) |
| **4D-Rotor Gaussian Splatting**                              | SIGGRAPH 2024          | 速度很强的 native 4D 表示；若论文声称实时或高帧率编辑/预览，应和这类渲染速度 baseline 对齐。 | 使用 anisotropic 4D XYZT Gaussians 与 temporal slicing；项目页报告 RTX4090 上 144 fps 或 583 fps 的高速渲染设置。([weify627.github.io](https://weify627.github.io/4drotorgs/)) | **完整可复现**。官方 GitHub 有 install、dataset、training、render、eval。([GitHub](https://github.com/weify627/4D-Rotor-Gaussians)) |
| **Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction** | CVPR 2024              | CTRL-D 等动态编辑方法的重要底座；canonical 3DGS + deformation field 是目前编辑传播的主流结构。 | 方法用 canonical 3D Gaussians 加 deformation field，并对时间和 pose 做平滑约束。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Deformable_3D_Gaussians_for_High-Fidelity_Monocular_Dynamic_Scene_Reconstruction_CVPR_2024_paper.html)) | **完整可复现**。官方 MIT 代码公开。([GitHub](https://github.com/ingra14m/Deformable-3D-Gaussians)) |
| **Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle** | CVPR 2024 Highlight    | 动态 Gaussian 粒子与显式运动建模，对动作编辑的运动场约束有借鉴价值。 | 使用 polynomial / Fourier Dynamic Deformation Distribution Module；论文摘要称训练比逐帧 3DGS 快约 5 倍，并优于既有方法。([arXiv](https://arxiv.org/abs/2312.03431)) | **基本可复现**。官方实现公开。([GitHub](https://github.com/NJU-3DV/Gaussian-Flow)) |
| **GaGS: Geometry-aware Gaussian Splatting for Scene Rendering** | CVPR 2024              | 几何感知动态 view synthesis baseline；适合作为“编辑前重建质量”和几何一致性的对照。 | 面向合成和真实动态场景，强调 geometry-aware dynamic view synthesis。([arXiv](https://arxiv.org/abs/2404.06270)) | **基本可复现**。官方代码公开，并包含 evaluation command。([GitHub](https://github.com/zhichengLuxx/GaGS)) |
| **4DGS in the Wild: A Novel View Synthesis Dataset of In-the-Wild Dynamic Scenes** | NeurIPS 2024           | 如果你要做真实单目/手持视频动态编辑，这类 in-the-wild 动态数据和不确定性建模需要纳入。 | 论文针对 casually recorded monocular videos，使用 uncertainty-aware regularization 与 dynamic densification 做动态场景重建。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e95da8078ec8389533c802e368da5298-Abstract-Conference.html)) | 开源信息需按具体仓库核验；作为数据/设定参考比作为动作编辑 baseline 更重要。 |
| **Splatter a Video: Video Gaussian Representation for Versatile Processing, VGR** | NeurIPS 2024           | 非标准 4DGS 编辑论文，但非常接近“动作/外观编辑”：把视频显式嵌入 3D Gaussians，并为每个 Gaussian 关联 3D motion，支持 tracking、depth、segmentation、view synthesis 和 motion / appearance editing。([OpenReview](https://openreview.net/forum?id=bzuQtVDxv0&referrer=[the+profile+of+XIAOJUAN+QI](%2Fprofile%3Fid%3D~XIAOJUAN_QI2))) | 训练信号包括 optical flow / depth distillation；任务覆盖 tracking、depth、segmentation、novel view synthesis、motion/appearance editing。([OpenReview](https://openreview.net/forum?id=bzuQtVDxv0&referrer=[the+profile+of+XIAOJUAN+QI](%2Fprofile%3Fid%3D~XIAOJUAN_QI2))) | **基本可复现**。官方代码公开。([GitHub](https://github.com/SunYangtian/Splatter_A_Video)) |
| **MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds** | CVPR 2025              | 动作编辑很需要显式 motion scaffold。MoSca 把动态场景表示为 4D motion scaffold，并把 Gaussians anchored 到 motion graph 上，适合借鉴为动作控制层。 | 面向 in-the-wild monocular video；同时优化 camera focal / pose bundle adjustment 和动态 Gaussian fusion。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.pdf)) | 项目和代码链接在论文中提供；可作为单目动态重建 / 运动骨架 baseline。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Lei_MoSca_Dynamic_Gaussian_Fusion_from_Casual_Videos_via_4D_Motion_CVPR_2025_paper.pdf)) |
| **Shape of Motion: 4D Reconstruction from a Single Video**   | ICCV 2025              | 对“动作变化幅度”很重要：它使用 persistent 3D motion trajectories 和 compact SE(3) motion bases，比纯 deformation MLP 更适合建模大幅运动。 | CVF 页面描述其从单视频重建通用动态场景，使用显式持久 3D motion trajectories 与 SE(3) motion bases；另有 canonical 3DGS、per-Gaussian motion coefficients、global motion bases。([CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Shape_of_Motion_4D_Reconstruction_from_a_Single_Video_ICCV_2025_paper.html)) | **基本可复现**。官方 GitHub 有训练命令和数据集说明。([GitHub](https://github.com/vye16/shape-of-motion/)) |
| **L4GM: Large 4D Gaussian Reconstruction Model**             | NeurIPS 2024           | 前馈式 4D Gaussian 的代表。若你要做“前馈式高斯 + 编辑”，L4GM 是 object-level 4DGS 生成/重建的必看 baseline。 | 从 single-view video 输入，在约 1 秒内输出 animated 3D Gaussians；训练数据为 Objaverse 44K objects、110K animations、12M videos、300M frames；基于 LGM 预测每帧 3D Gaussian，并插值到高 FPS。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6808f2c57d9564a2639a4710e3bbd9b9-Abstract-Conference.html)) | **较完整可复现**。官方 GitHub Apache-2.0，包含 model weights 和 Gradio demo。([GitHub](https://github.com/nv-tlabs/L4GM-official)) |
| **4D Gaussian Transformer, 4DGT**                            | NeurIPS 2025 Spotlight | 目前更贴近真实视频场景的前馈式 4D Gaussian baseline。若你要做“实时/前馈编辑初始化”，它比 L4GM 更接近 real-world monocular dynamic scenes。 | 前馈式 4D Gaussian Transformer；64-frame rolling window；从 hours-to-seconds；项目页列出 ADT、HOT3D、Nymeria、DyCheck、TUM Dynamics，并与 L4GM、Shape-of-Motion、StaticLRM 等比较。([arXiv](https://arxiv.org/abs/2506.08015)) | **部分到基本可复现**。GitHub 有 configs、data、docs、pretrained model、inference；但 README 说明 full evaluation / inference on other datasets still work in progress。([GitHub](https://github.com/facebookresearch/4dgt)) |
| **DrivingRecon: Large 4D Gaussian Reconstruction Model for Autonomous Driving** | NeurIPS 2025           | domain-specific，但对“前馈式 4DGS + 大场景 + 动态物体 + 编辑应用”有参考意义。 | 面向 autonomous driving 的 feed-forward 4D reconstruction；提出 PD-Block 处理 dynamic/static decoupling；OpenReview 摘要提到应用包括 scene editing。([OpenReview](https://openreview.net/forum?id=HF5A73jmxq&referrer=[the+profile+of+Kurt+Keutzer](%2Fprofile%3Fid%3D~Kurt_Keutzer1))) | **基本可复现**。官方仓库包含 getting started、environment、dataset prep、training/eval。([GitHub](https://github.com/EnVision-Research/DriveRecon)) |
| **CAT4D: Create Anything in 4D with Multi-View Video Diffusion Models** | CVPR 2025              | 不是编辑 baseline，但对“生成缺失视角/新动作幅度”很有启发：先用多视角视频 diffusion 生成 dense multi-view videos，再优化 deformable 3D Gaussian。 | 从 monocular video 生成 multi-view videos，再重建 dynamic 3D scene；CVPR 页面和项目页均强调 deforming 3D Gaussians 与 camera/time disentanglement。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_CAT4D_Create_Anything_in_4D_with_Multi-View_Video_Diffusion_Models_CVPR_2025_paper.pdf)) | 开源状态需单独核验；作为生成式先验/数据增强路线重要。        |
| **Feature4X: Bridging Any Monocular Video to 4D Agentic AI with Versatile Gaussian Feature Fields** | CVPR 2025              | 如果动作编辑需要语言定位、部件选择、可交互控制，4D feature field 是必要工具链。 | 以 4D Gaussian feature field 连接 2D foundation models，展示 novel view segment-anything、geometric / appearance editing、free-form VQA。([CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Feature4X_Bridging_Any_Monocular_Video_to_4D_Agentic_AI_with_CVPR_2025_paper.html)) | **基本可复现**。官方代码含 agent editing、interactive、feature distillation、setup scripts。([GitHub](https://github.com/ShijieZhou-UCLA/Feature4X)) |

## 3. 进入该领域必须复现/对比的 SOTA baseline

如果你的目标是普通“文本驱动 4DGS 编辑”，最小 baseline 集合应是：

1. **Instruct 4D-to-4D**：虽然不是 4DGS，但它是 4D text editing 的早期标准 baseline，后续 Instruct-4DGS、CTRL-D、Catalyst4D、Dynamic-eDiTor 都绕不开。
2. **Instruct-4DGS**：4DGS 静动分离外观编辑 baseline；速度和效率必须对比。
3. **CTRL-D**：局部可控 dynamic 3D scene editing baseline，尤其适合“用户给单张参考编辑”的设置。
4. **Dynamic-eDiTor / Catalyst4D**：如果你投稿时间在 2026 之后，且其代码或补充材料可用，这两者应进入主对比；前者偏 MM-DiT + 4DGS 直接优化，后者偏 3D-to-4D propagation。
5. **Control4D**：只在 portrait / human head / dynamic portrait 设定中必须对比；通用动态动作编辑不建议把它作为核心 baseline，因为它领域过窄且代码未完整释放。

如果你的目标是“动作编辑准确性、连贯性、动作变化幅度”，最小 baseline 集合应改成：

1. **SC-GS**：稀疏控制点 + 4DGS，是最直接的可控动作/几何编辑 baseline。
2. **D-MiSo**：mesh-like Gaussian soup，是目前更直接面向 dynamic 3D scene editing 的 NeurIPS 2024 baseline。
3. **VGR / Splatter a Video**：视频 Gaussian 表示里显式带 3D motion，适合作为 video-to-3D motion editing 对照。
4. **MoSca**：如果输入是 casual monocular video，它的 4D motion scaffold 很适合作为 motion representation baseline。
5. **Shape-of-Motion**：如果你强调大幅动作、持久轨迹、SE(3) motion bases，它比一般 deformation MLP 更应该被对比。
6. **4DGS / Deformable 3DGS / 4D-Rotor / SpacetimeGS**：作为编辑前的重建和实时渲染底座对比，不一定都是编辑 baseline，但必须证明你的编辑方法没有显著损坏重建质量和 FPS。

如果你的目标是“前馈式高斯 + 快速编辑”，必须对比：

1. **L4GM**：object-level 前馈 4D Gaussian reconstruction。
2. **4DGT**：real-world monocular dynamic scene 的前馈 4D Gaussian Transformer。
3. **DrivingRecon**：仅当你的场景涉及自动驾驶或大规模街景。
4. **CAT4D**：不是严格编辑 baseline，但如果你使用 video diffusion / multiview generation prior，它是非常强的相关生成式路线。

## 4. 必须使用的数据集

通用 4D / 4DGS 编辑建议至少覆盖四类数据，否则论文说服力不足。

第一类是 **DyNeRF / N3DV**。这是 Instruct-4DGS、CTRL-D、Dynamic-eDiTor、Catalyst4D 都使用或涉及的核心动态多视角数据。建议至少包含 coffee_martini、sear/flame steak、cut_roasted_beef、cook spinach、flame salmon 这类有明显物体运动、手部交互或非刚体变化的场景。

第二类是 **DyCheck / HyperNeRF**。这类数据更接近 casual / monocular / in-the-wild 动态场景，是 CTRL-D、Instruct 4D-to-4D、Catalyst4D、Dynamic-eDiTor 相关论文中的重要测试域。它可以检验方法在非理想多视角采集和复杂动态中的稳定性。

第三类是 **D-NeRF、NeRF-DS、PanopticSports**。如果你强调动作编辑，这三类比纯 DyNeRF 外观编辑更关键。D-NeRF 适合可控合成运动；NeRF-DS 适合真实动态和复杂反射/形变；PanopticSports 适合人体/运动动作变化。SC-GS 和 D-MiSo 都使用 D-NeRF / NeRF-DS，D-MiSo 还使用 PanopticSports。([arXiv](https://arxiv.org/html/2405.14276v3))

第四类是前馈式和大规模训练数据。若做 object-level 前馈 4DGS，需要 **Objaverse-4D / L4GM 数据设置**；若做真实单目视频前馈，需要关注 **ADT、HOT3D、Nymeria、DyCheck、TUM Dynamics** 等 4DGT 使用的数据；若做驾驶场景，则进入 DrivingRecon / autonomous driving 数据域。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6808f2c57d9564a2639a4710e3bbd9b9-Abstract-Conference.html))

## 5. 必做实验与指标

仅报告 CLIP similarity 和用户研究不够，尤其不够支撑“动作编辑准确性、连贯性、变化幅度”。建议把实验拆成五组。

第一组是编辑前重建质量。必须报告 held-out novel view / novel time 的 **PSNR、SSIM、LPIPS**，并报告训练时间、显存、模型大小、渲染 FPS。否则很难判断编辑提升是否来自牺牲底层重建质量。

第二组是外观/语义编辑质量。沿用已有论文的 **CLIP text-image similarity、CLIP directional similarity、EditScore / VBench subject consistency / DINO consistency**，并给出多视角和多时间平均值。CTRL-D、Dynamic-eDiTor、Catalyst4D 这条线都使用了 CLIP、VBench/DINO、temporal consistency 或类似指标。([ar5iv](https://ar5iv.org/abs/2412.01792))

第三组是时序一致性。建议至少报告 **RAFT warping error、tLPIPS、flicker score、DINO temporal consistency**。Dynamic-eDiTor 已使用 RAFT warping 和 MEt3R 等指标，这应成为 2026 后论文的强制项。([arXiv](https://arxiv.org/html/2512.00677v1))

第四组是多视角一致性。必须报告 cross-view consistency，例如 **MEt3R multi-view consistency、跨视角 mask IoU、跨视角 feature consistency、reprojection error**。如果动作编辑后不同相机看到的物体位置不一致，CLIP 分数再高也没有意义。

第五组是动作编辑专用指标。这是当前文献的缺口，也是你最可能做出突破的地方。建议定义一套 action-edit benchmark，包括：

- **3D trajectory error**：编辑目标点/控制点/骨架关节的 3D endpoint error、平均轨迹误差、ATE/RPE。
- **2D multi-view keypoint error**：每个相机视角下的 keypoint EPE、PCK、mask IoU。
- **动作幅度曲线**：小/中/大位移，小/中/大旋转，速度变化、时间反转、phase shift；画出动作幅度 vs 质量退化曲线。
- **连续性与物理合理性**：jerk / acceleration smoothness、contact preservation、penetration rate、ARAP energy 或局部刚性误差。
- **局部保持**：未编辑区域 PSNR/LPIPS、background leakage、identity preservation。
- **失败率**：遮挡、新显露区域、拓扑变化、快速运动、多人交互分别统计。

## 6. 对研究切入点的判断

如果你想在“动作编辑准确性、连贯性、动作变化幅度”上突破，不建议沿着“给 4DGS 接一个 InstructPix2Pix，然后用 CLIP / SDS 修外观”的路线继续微调。这条路线的强项是 appearance/style/object attribute editing，不是大幅动作控制。Instruct-4DGS 已经明确把 motion editing 排除在主要能力之外，CTRL-D 和 Catalyst4D 也主要围绕外观、局部内容和 3D-to-4D 传播。([ar5iv](https://ar5iv.org/html/2502.02091v3))

更合理的路线是把问题拆成三层。

第一层是 **显式 4D motion control representation**。可从 SC-GS 的 sparse control points、D-MiSo 的 Multi-Gaussians Soup、MoSca 的 motion scaffold、Shape-of-Motion 的 SE(3) motion bases 中吸收结构。目标不是让 diffusion 直接“想象动作”，而是让动作成为可优化、可度量、可约束的 3D/4D 变量。

第二层是 **动作条件生成/传播先验**。语言 prompt 可以转成目标轨迹、关键帧、骨架姿态、接触约束或对象位姿序列；video diffusion / MM-DiT 可以作为先验，但必须被 3D trajectory、multi-view reprojection 和 temporal consistency 约束住。否则容易得到语义上像、几何上错、时间上漂的结果。

第三层是 **快速初始化 + 局部 4D refinement**。如果你要追求实时或准实时编辑，不应每次从头优化 4DGS。更可行的是用 L4GM / 4DGT 这类前馈式模型给出 4D Gaussian 初始化，再对局部控制区域做低秩 motion basis 或 sparse anchor refinement。这里要明确区分“实时渲染”和“实时编辑”：目前强编辑方法的耗时仍多在几十分钟级别，而实时 4DGS 论文解决的主要是 rendering FPS，不是 edit latency。

最终建议的最小论文实验包是：

- baseline：SC-GS、D-MiSo、Instruct 4D-to-4D、Instruct-4DGS、CTRL-D、4DGS / Deformable 3DGS；若 2026 投稿，再加入 Dynamic-eDiTor、Catalyst4D、4DGT。
- dataset：DyNeRF/N3DV、DyCheck/HyperNeRF、D-NeRF、NeRF-DS、PanopticSports；若做前馈，再加入 Objaverse-4D 或 4DGT 的 real-world video 数据。
- 指标：PSNR/SSIM/LPIPS/FPS + CLIP/EditScore + temporal consistency + multi-view consistency + action trajectory/keypoint/amplitude metrics。
- 消融：无显式 motion control、无 temporal propagation、无 multi-view consistency、无局部保持损失、不同动作幅度、不同遮挡程度、不同编辑区域大小、不同输入视角数量。