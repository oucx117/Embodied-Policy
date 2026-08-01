# Embodied Policy 经典工作

本仓库用于整理关于具身智能策略的经典工作，重点关注 VA、VLA 和 WAM 三类 Embodied Policy 建模范式。

## 目录

- [分类框架](#classification)
- [仓库结构](#repository-structure)
- [VA](#va)
- [VLA](#vla)
- [WAM](#wam)
- [声明](#notice)

<a id="classification"></a>

## 分类框架

> 按 Policy 的输入-输出建模范式划分

| 大类                             | 建模范式                           | 核心问题                                                     |
| -------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| **VA** (Vision-Action)           | $p(a_{t+1:t+k} \mid o_t)$                  | 不依赖语言，根据视觉观测直接生成动作                         |
| **VLA** (Vision-Language-Action) | $p(a_{t+1:t+k} \mid o_t, l)$               | 利用 VLM 的语义知识，根据语言指令泛化到新任务、新物体、新场景 |
| **WAM** (World-Action Model)     | $p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)$ | 预测未来状态（通常以视频方式表现）和物理动作                 |

<a id="repository-structure"></a>

## 仓库结构

```text
.
├── VA/    # Vision-Action
├── VLA/   # Vision-Language-Action
├── WAM/   # World-Action Model
├── Method/ # 一些通用方法/模块
└── README.md
```

> 每个子目录通常包含对应论文笔记和论文 PDF

<a id="va"></a>

## VA

> 研究动机：真实机器人控制需要**高频、稳定、连续**的低层动作输出。
> 应用场景：本身**不具备泛化性**，更多适用于**单一、高精度 manipulation 任务**。

| 时间 | 方法 | 论文 | 简介 |
| --- | --- | --- | --- |
| 2023.3 | [Diffusion Policy](./VA/Diffusion_policy/Diffusion_policy.md) | [2303.04137](https://arxiv.org/abs/2303.04137) | 条件扩散：迭代去噪生成多模态连续动作 |
| 2023.4 | [ACT](./VA/ACT/ACT.md) | [2304.13705](https://arxiv.org/abs/2304.13705) | 动作分块：一次并行预测未来一段动作序列 |

<a id="vla"></a>

## VLA

> 研究动机：单纯 VA Policy 通常不具备泛化性，只能应对单个任务，无法支持新指令、新物体和新任务组合；而 VLM 已经从互联网规模数据中学到了**丰富的语义知识、物体概念、空间关系和指令理解能力**，因此可以把**这些知识迁移到机器人控制中**，提升泛化性。

| 时间 | 方法 | 论文 | 简介 |
| --- | --- | --- | --- |
| 2022.12 | [RT-1](./VLA/RT系列/RT1.md) | [2212.06817](https://arxiv.org/abs/2212.06817) | 规模化训练：用单一高容量模型学习多任务真实机器人控制 |
| 2023.7 | [RT-2](./VLA/RT系列/RT2.md) | [2307.15818](https://arxiv.org/abs/2307.15818) | VLA 范式：将动作表示为文本 token，在 VLM 上联合微调 |
| 2024.6 | [OpenVLA](./VLA/OpenVLA/OpenVLA.md) | [2406.09246](https://arxiv.org/abs/2406.09246) | 开源 VLA：在预训练 VLM 上用跨实体机器人数据训练 |
| 2024.10 | [RDT-1B](./VLA/RDT/RDT.md) | [2410.07864](https://arxiv.org/abs/2410.07864) | 统一动作空间：跨机器人预训练双臂扩散策略 |
| 2024.10 | [π0](./VLA/π系列/π0.md) | [2410.24164](https://arxiv.org/abs/2410.24164) | 动作专家：以 Flow Matching 并行生成高频连续动作 |
| 2025.1 | [FAST](./VLA/π系列/FAST.md) | [2501.09747](https://arxiv.org/abs/2501.09747) | 动作压缩：结合 DCT、量化和 BPE 生成高效动作 token |
| 2025.2 | [HAMSTER](./VLA/HAMSTER/HAMSTER.md) | [2502.05485](https://arxiv.org/abs/2502.05485) | 层级控制：以 2D 轨迹连接 VLM 规划与低层策略 |
| 2025.2 | [OpenVLA-OFT](./VLA/OpenVLA/OpenVLA-oft.md) | [2502.19645](https://arxiv.org/abs/2502.19645) | 优化微调：并行解码连续动作以提升性能和控制频率 |
| 2025.3 | [GR00T N1](./VLA/GR00T系列/GR00T_N1.md) | [2503.14734](https://arxiv.org/abs/2503.14734) | 快慢双系统：结合 VLM 语义理解与连续动作生成 |
| 2025.4 | [π0.5](./VLA/π系列/π05.md) | [2504.16054](https://arxiv.org/abs/2504.16054) | 异构数据联合训练：提升开放世界长程任务泛化能力 |
| 2025.8 | [MolmoAct](./VLA/MolmoAct/MolmoAct.md) | [2508.07917](https://arxiv.org/abs/2508.07917) | 动作推理：依次生成深度、视觉轨迹和动作 token |
| 2025.8 | [Embodied-R1](./VLA/Embodied-R1/Embodied-R1.md) | [2508.13998](https://arxiv.org/abs/2508.13998) | 具身指向：用形态无关的中间表示连接推理与执行 |
| 2025.9 | [PEEK](./VLA/PEEK/PEEK.md) | [2509.18282](https://arxiv.org/abs/2509.18282) | 关键点引导：分离高层视觉推理与低层动作执行 |
| 2025.11 | [π*0.6](./VLA/π系列/π06.md) | [Technical Report](https://www.pi.website/download/pistar06.pdf) | 经验学习：用人工干预和优势加权 Flow Matching 强化策略 |
| 2026.2 | [VLA-JEPA](./VLA/VLA-JEPA/VLA-JEPA.md) | [2602.10098](https://arxiv.org/abs/2602.10098) | 潜在世界模型：从无动作标签视频学习状态转移表示 |
| 2026.2 | [Xiaomi-Robotics-0](./VLA/Xiaomi-Robotics系列/Xiaomi-Robotics-0.md) | [2602.12684](https://arxiv.org/abs/2602.12684) | 异步执行：兼顾连续动作生成与实时环境响应 |
| 2026.4 | [π0.7](./VLA/π系列/π07.md) | [Technical Report](https://www.pi.website/download/pi07.pdf) | 多模态提示：从异质数据中学习可操纵的组合泛化策略 |
| 2026.5 | [Qwen-VLA](./VLA/Qwen-VLA/Qwen-VLA.md) | [2605.30280](https://arxiv.org/abs/2605.30280) | 统一动作轨迹空间：覆盖操作、导航和人类第一视角任务 |
| 2026.6 | [Qwen-RobotManip](./VLA/Qwen-RobotManip/Qwen-RobotManip.md) | [2606.17846](https://arxiv.org/abs/2606.17846) | 对齐后扩展：统一跨本体、坐标系和行为分布的数据 |
| 2026.7 | [Xiaomi-Robotics-1](./VLA/Xiaomi-Robotics系列/Xiaomi-Robotics-1.md) | [2607.15330](https://arxiv.org/abs/2607.15330) | 规模化 UMI 预训练：用超 10 万小时真实轨迹学习通用操作先验 |

<a id="wam"></a>

## WAM

> 研究动机：VLA 通常是 reactive mapping，即从当前观测和语言直接输出动作 $p(a\mid o,l)$ ，但不显式建模“如果执行某个动作，世界会如何变化”。这在长时程任务、接触丰富任务、遮挡场景、失败恢复和物理常识推理中会受限。WAM 定义为联合预测未来状态和动作的 embodied foundation model，即建模 $p(o',a\mid o,l)$ ，目标是让 Policy 具备 physical foresight。

| 时间 | 方法 | 论文 | 简介 |
| --- | --- | --- | --- |
| 2025.12 | [Motus](./WAM/Motus/Motus.md) | [2512.13030](https://arxiv.org/abs/2512.13030) | 潜在动作：统一视频、语言和机器人动作的多种建模模式 |
| 2026.1 | [Cosmos Policy](./WAM/CosmosPolicy/CosmosPolicy.md) | [2601.16163](https://arxiv.org/abs/2601.16163) | 潜在帧注入：直接微调视频模型进行控制和价值规划 |
| 2026.1 | [LingBot-VA](./WAM/LingBot-VA/LingBot-VA.md) | [2601.21998](https://arxiv.org/abs/2601.21998) | 因果世界建模：自回归交错生成未来视频与动作块 |
| 2026.2 | [DreamZero](./WAM/DreamZero/Dream-Zero.md) | [2602.15922](https://arxiv.org/abs/2602.15922) | 视频动作联合生成：继承视频模型先验实现零样本控制 |
| 2026.3 | [Fast-WAM](./WAM/Fast-WAM/Fast-WAM.md) | [2603.16666](https://arxiv.org/abs/2603.16666) | 训练时建模视频、推理时跳过未来生成以降低延迟 |
| 2026.6 | [ImageWAM](./WAM/ImageWAM/ImageWAM.md) | [2606.19531](https://arxiv.org/abs/2606.19531) | 图像编辑先验：以中间 KV 表征替代完整未来视频生成 |

<a id="notice"></a>

## 声明

本仓库仅用于个人学习和论文阅读整理。论文、图片及相关材料的版权归原作者、论文出版方或项目团队所有。
