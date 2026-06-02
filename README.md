# Embodied Policy 框架学习

### 📂 按 Policy 的输入-输出建模范式划分大方向

| 大类                            | 建模范式                                                     | 核心问题                                                     |
| ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **VA**：Vision-Action           | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t)$                  | 不依赖语言，**根据视觉观测直接生成动作**                     |
| **VLA**：Vision-Language-Action | $p(\mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$            | 利用 VLM 的语义知识，根据语言指令**泛化**到新任务、新物体、新场景 |
| **WAM**：World-Action Model     | $p(\mathbf{o}_{t+1:t+k}, \mathbf{a}_{t+1:t+k} \mid \mathbf{o}_t, \ell)$ | 预测**未来状态（通常以视频方式表现）**和物理动作             |

---

### 📚 VA：从视觉到动作的 Policy

> 研究动机：真实机器人控制需要**高频、稳定、连续**的低层动作输出。
>
> 应用场景：本身**不具备泛化性**，更多适用于**单一、高精度 manipulation 任务**。

#### 📖 代表工作

* **ACT**：全称是 **A**ction **C**hunking **T**ransformer。核心思想是 action chunking（动作分块），即 policy **一次预测未来一段动作序列**，而不是每个时刻只预测单步动作，有效克服误差累积，并结合时间集成技术提升动作平滑性。模型架构上采用CVAE。

  ![model](./VA/ACT/images/model.png)

* **Diffusion_policy**：借鉴视觉生成领域的 Diffusion 机制，从纯高斯噪声开始，在视觉观测的“条件引导”下，通过**多步迭代去噪**，生成一段连续的机器人动作序列。Diffusion 机制天生具备**强大的多模态生成能力和极其稳定的训练过程**，有利于解决人类示教数据存在的多模态性（例如面对同一障碍物，人类有时从左边绕，有时从右边绕）。

  ![model](./VA/Diffusion_policy/images/model.png)

---

### 📚 VLA：从语言、视觉到动作的通用机器人 policy

> 研究动机：单纯 VA policy 通常不具备泛化性，只能应对单个任务，无法支持新指令、新物体和新任务组合；而 VLM 已经从互联网规模数据中学到了**丰富的语义知识、物体概念、空间关系和指令理解能力**，因此可以把**这些知识迁移到机器人控制中**，提升泛化性。

#### 📖 代表工作

##### Autoregressive VLA（自回归 VLA）

> 早期的 VLA 通常继承传统大模型的**自回归生成机制**，代表工作如 RT-2、OpenVLA 等。这类工作的核心思想是**把机器人动作离散化（通常是分成 256 个 bin）为类似语言 token 的形式，让 VLM 可以像生成文本一样生成动作**。

* **RT-2**：直接在强大的预训练 VLM（PaLI-X 5B/55B 和 PaLM-E 12B）上进行**联合微调**（机器人数据 + 原始的网络图文数据），将 VLM 在互联网规模数据上学到的丰富语义知识和推理能力迁移到机器人的底层控制中，得到端到端的 VLA 模型。

  ![model](./VLA/RT系列/images/model.png)

* **OpenVLA**：同 RT2 一样直接继承强大的预训练 VLM（Prismatic-7B）训练得到。架构上采用**融合视觉编码器**，由 SigLIP 和 DINOv2 分别提取粗粒度和细粒度信息。模型完全**开源**，解决了此前 VLA 多为闭源、难以复现的问题。

  ![model](./VLA/OpenVLA/images/model.png)

##### Diffusion / Flow Matching based VLA

> 目前大部分的 VLA 模型通常采用**基于 diffusion 或 flowmatching 机制**来输出动作，相比于自回归方式，基于 diffusion 或 flowmatching 机制能够**并行**输出**连续**动作，有助于提高动作执行频率和精度，使得能够完成高度灵巧的任务。

* **$\pi_0$**：MoT架构（预训练 VLM + **Flow Matching Action Expert**），使用预训练的 VLM 提取视觉和语言特征作为 Condition，然后通过 **Flow Matching** 生成连续动作，极大提升了机器人动作频率（50Hz）和精度，使得能够完成高度灵巧的任务。

  ![pi0_model](./VLA/π系列/images/pi0_model.png)

* **RDT**：面向双臂 manipulation，提出**统一动作空间**（128维向量空间，预先划分好了各种可能用到的专属物理量槽位），使得模型能够打破物理形态壁垒，进行**跨机器人实体预训练**，从而学习可广泛迁移的物理控制先验知识。

  ![model](./VLA/RDT/images/model.png)

