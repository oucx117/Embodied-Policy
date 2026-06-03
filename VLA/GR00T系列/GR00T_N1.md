## GR00T N1: An Open Foundation Model for Generalist Humanoid Robots

### 一. 工作动机

**核心问题**：通用机器人不仅需要能够适应人类环境的硬件形态，还需要能够理解开放式任务、处理真实世界变化并快速学习新技能的基础模型。已有 VLA / imitation learning 方法通常受限于某一类机器人本体、某一类数据来源或某一类任务，难以直接扩展到**通用 humanoid robot**。更关键的是，真实 humanoid 数据采集成本极高，不存在类似互联网文本/图像那样的大规模 humanoid robot 数据，因此单靠某一种机器人或某一种真实数据很难训练出真正的泛化模型。

**核心思想**：GR00T N1 将 humanoid robot foundation model 建模为一个 Vision-Language-Action (VLA) 模型，并采用“**快慢系统（Dual-System）**”架构：慢系统 **System 2** 使用预训练 VLM 处理视觉和语言，负责语义理解与任务解释；快系统 **System 1** 使用基于 Flow Matching 的 Diffusion Transformer 生成高频连续动作，负责实时闭环控制。与此同时，GR00T N1 用一个“**数据金字塔**”组织训练数据：底层是大规模 web data / human videos，中层是 simulation data 和 neural generated trajectories，顶层是真实机器人轨迹，从而在规模、泛化性和真实执行 grounding 之间取得平衡。

---

### 二. GR00T N1 模型

GR00T N1 是一个面向通用 humanoid robot 的 VLA 模型，输入为图像观测、语言指令和机器人状态，输出为连续的动作。公开模型 **GR00T-N1-2B** 总参数量约 **2.2B**，其中 VLM 部分约 **1.34B**。模型一次生成长度为 $H=16$ 的动作块，在 NVIDIA L40 GPU 上使用 bf16 推理一个动作块的时间约为 **63.9ms**。

![model](./images/model.png)

**A. 整体架构**

**双系统架构**是 GR00T N1 的核心设计。它不是让一个单一 Transformer 同时承担语义理解和低层动作生成，而是将模型明确拆分为两个紧密耦合、端到端联合训练的模块。

- **System 2：Vision-Language Module**
  - 使用预训练的 **Eagle-2 VLM** 作为视觉语言主干。
    - 输入包括一个或多个图像观测以及语言任务描述。
    - 输出视觉语言 token embeddings，用作动作生成模块的条件信息。
  - 主要承担任务理解、视觉语义解析、目标识别和高层上下文建模。

- **System 1：Diffusion Transformer Module**
  - 使用 DiT-style Transformer，并通过 **action flow matching** 生成连续动作。
    - 输入包括机器人本体状态、带噪动作块、diffusion / flow matching 时间步，以及来自 VLM 的视觉语言特征（通过 Cross-Attention 注入）。
    - 输出去噪后的动作块。
  - 主要承担高频闭环运动生成和精细动作控制。

> 这里的“快慢系统”可以理解为：System 2 负责**较慢但语义更强的视觉语言理解**，System 1 负责**较快且连续的动作生成**。论文中提到 System 2 在 L40 GPU 上约以 10Hz 运行，而 System 1 可以生成 120Hz 的闭环 motor actions。

**B. 视觉语言模块（System 2）**

GR00T N1 使用 **Eagle-2 VLM** 编码视觉和语言输入。Eagle-2 由 SmolLM2 LLM 和 SigLIP-2 图像编码器组成，并在大规模视觉语言任务上完成对齐。

- **图像输入**：
  - 图像分辨率为 $224 \times 224$。
  - 图像经过 SigLIP-2 编码后，再经过 pixel shuffle。
  - 每帧图像最终得到 **64 个 image token embeddings**。

- **语言输入**：
  - 任务描述以 VLM chat format 输入。
  - 例如：`Pick up the apple and place it into the bottom shelf`。
  - 语言 token 与图像 token 一起进入 VLM 进行上下文建模。

- **特征提取**：
  - GR00T N1 不直接使用 VLM 的最终层输出，而是使用中间层表示。
  - 对 GR00T-N1-2B，论文使用 **第 12 层**的 VLM 表示。
  - 这样做同时带来更快的推理速度和更高的下游 policy success rate。

> 这与很多 VLA 模型类似：VLM 不直接输出动作，而是先把图像和语言转换成强语义条件，再交给动作模块生成连续控制信号。

**C. 状态与动作编码层**

GR00T N1 需要支持从单臂机械臂、双臂机械臂到 humanoid robot 的多种 embodiment，因此不同机器人的 state / action 维度并不一致。论文没有采用简单的统一零填充方案，而是**为不同 embodiment 设计了专属的状态编码器和动作编码器**。

- **状态编码器（State Encoder）**：
  - 每个 embodiment 使用一个 MLP。
  - 将该 embodiment 的机器人状态 $q_t$ 投影到**统一的 DiT embedding 空间**。
  - 状态可以包括 joint positions、joint velocities、base position、EEF poses 等。

- **动作编码器（Action Encoder）**：
  - 每个 embodiment 使用一个 MLP。
  - 输入为带噪动作向量和 flow matching 时间步。
  - 将 noised action chunk $A_t^\tau$ 编码为动作 token 序列。
  - 与 $\pi_0$ 类似，动作生成采用 action chunking，一次处理：
    $$
    A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]
    $$
    其中 GR00T N1 设置 $H=16$。

- **动作解码器（Action Decoder）**：
  - 每个 embodiment 使用专属 MLP 解码器。
  - 将 DiT 的最终动作 token 输出映射回该 embodiment 的实际动作空间。

> 这种 embodiment-specific encoder / decoder 的作用是：在中间层共享通用策略表示，但在输入输出端保留不同机器人本体的结构差异，从而实现跨 embodiment 训练。

**D. Diffusion Transformer 动作模块（System 1）**

GR00T N1 的动作模块是一个基于 DiT 的 flow-matching policy。它通过交替的 self-attention 和 cross-attention block 同时处理动作、状态与视觉语言条件。

- **Self-Attention**：
  - 作用在 noised action token embeddings $A_t^\tau$ 和 state embeddings $q_t$ 上。
  - 负责建模动作块内部的时序关系以及动作与本体状态之间的耦合。

- **Cross-Attention**：
  - 让动作 token attend 到 VLM 输出的视觉语言 token embeddings $\phi_t$。
  - 负责把任务语义、目标物体、空间关系等高层信息注入到低层动作生成中。

- **输出层**：
  - DiT 最终输出 $H$ 个动作 token。
  - embodiment-specific Action Decoder 将这些 token 转换成对应机器人的 motor actions。

> 与 $\pi_0$ 中“单一 Transformer + Action Expert”的设计不同，GR00T N1 更明确地采用 **VLM + DiT** 的组合式结构：VLM 作为 System 2 提供语义条件，DiT 作为 System 1 负责动作去噪和运动生成。

---

### 三. 训练与推理

**训练流程**：GR00T N1 使用 action flow matching 训练动作生成模块，并将多种数据来源统一成“状态、图像、语言 $\rightarrow$ 动作”的监督格式。

1. **取真实动作块**：从数据集中采样观测、语言指令、机器人状态和真实动作块 $(\phi_t, q_t, A_t)$；
2. **采样时间步 $\tau$**：从设定的 Beta 分布中采样 flow matching 时间步 $\tau \in [0,1]$；
3. **加噪**：采样高斯噪声 $\epsilon \sim \mathcal{N}(0,I)$，构造带噪动作块：
   $$
   A_t^\tau = \tau A_t + (1-\tau)\epsilon
   $$
4. **模型预测向量场**：将视觉语言特征 $\phi_t$、机器人状态 $q_t$ 和带噪动作块 $A_t^\tau$ 输入 DiT，得到预测向量场：
   $$
   V_\theta(\phi_t, A_t^\tau, q_t)
   $$
5. **计算 Flow Matching 损失**：模型目标是拟合从带噪动作到真实动作的 denoising vector field，论文中的损失为：
   $$
   \mathcal{L}_{fm}(\theta)=\mathbb{E}_{\tau}\left[\left\|V_\theta(\phi_t,A_t^\tau,q_t)-(\epsilon-A_t)\right\|^2\right]
   $$
6. **联合优化**：VLM 与 DiT 紧密耦合训练。训练细节中，DiT、视觉编码器和 embodiment-specific adapter 模块会参与训练；语言相关部分保持冻结或基本不更新（LLM主干），以保留预训练 VLM 的语言能力。

**推理流程**：推理时，GR00T N1 从随机噪声动作块开始，通过少量 Euler integration steps 逐步去噪生成动作。

1. **编码观测与语言**：图像和语言输入经过 Eagle-2 VLM，得到视觉语言特征 $\phi_t$；
2. **编码机器人状态**：机器人状态 $q_t$ 经过 embodiment-specific State Encoder；
3. **初始化动作噪声**：
   $$
   A_t^0 \sim \mathcal{N}(0,I)
   $$
4. **K 步去噪**：使用 forward Euler integration 更新动作块：
   $$
   A_t^{\tau + 1/K}=A_t^\tau+\frac{1}{K}V_\theta(\phi_t,A_t^\tau,q_t)
   $$
5. **输出动作块**：经过 $K$ 步后输出长度为 $H=16$ 的动作块。论文中发现 $K=4$ 在多种 embodiment 上效果较好。

---

### 四. 数据构建

GR00T N1 的数据策略可以概括为 **Data Pyramid for Robot Foundation Model Training**。核心思想不是把所有数据简单混在一起，而是按数据规模与 embodiment-specific 程度组织为三层。

**A. 数据金字塔**

- **底层：Web Data & Human Videos**
  - 数据规模最大，但不直接包含机器人动作。
  - 提供丰富的视觉语义、人类行为、物体 affordance 和自然运动先验。
  - 使用 Ego4D、Ego-Exo4D、Assembly-101、EPIC-KITCHENS、HOI4D、HoloAssist、RH20T-Human 等人类视频数据。

- **中层：Synthetic Data**
  - 包括 simulation trajectories 和 neural generated trajectories。
  - simulation data 提供可自动扩展、带真实动作标签的机器人轨迹。
  - neural trajectories 通过 image-to-video model 生成 counterfactual robot videos（通过视频生成模型“想象”机器人操作过程），用于扩展真实数据的多样性。

- **顶层：Real-World Data**
  - 规模最小但最接近真实部署。
  - 包括 GR-1 humanoid teleoperation data、Open X-Embodiment、AgiBot-Alpha 等。
  - 提供真实机器人执行时的本体状态、动作和传感器观测。

**B. Latent Actions**

人类视频和 neural trajectories 通常没有真实机器人动作标签。为了解决 action-less data 无法用于 VLA 训练的问题，GR00T N1 使用 latent action learning。

- 使用 VQ-VAE 从视频帧对 $(x_t, x_{t+H})$ 中学习 latent action $z_t$。
  - 编码器根据当前帧和未来帧提取隐式动作表示。
  - 解码器根据当前帧和 latent action 重建未来帧。

- 训练完成后，将编码器作为一种 inverse dynamics model 使用，从视频中抽取 latent action label。
- 这些 latent actions 被当作一种特殊的 embodiment 参与预训练。

> 直观理解：latent action 不一定对应真实机器人的关节控制量，但它能表示“画面从当前状态变化到未来状态所隐含的动作”，因此可以把大量无动作标签的人类视频转化为可用于策略学习的监督信号。

**C. Neural Trajectories**

真实机器人数据采集成本高，而 video generation model 可以根据初始帧和新语言指令生成 counterfactual trajectories。GR00T N1 将 image-to-video model 在 88 小时的 in-house GR-1 teleoperation data 上微调，然后生成约 **827 小时**的 neural robot videos，相当于将真实数据扩展约 **10 倍**。

- 先用多模态 LLM 根据初始帧识别物体。
- 自动生成新的可行动作组合，例如 `pick up {object} from {location A} to {location B}`。
- 使用 video generation model 生成新轨迹。
- 再用多模态 LLM 过滤不符合语言指令的视频，并进行 re-captioning。
- 对生成视频使用 latent action 或 IDM-based pseudo-action 打标签。

> neural trajectories 的作用不是替代真实数据，而是在真实数据稀缺时提供更丰富的物体、目标位置、初始状态和语言指令组合。

**D. Simulation Trajectories**

GR00T N1 使用 DexMimicGen 在仿真中自动扩展少量人类示教。其基本流程是把示教轨迹分解成 object-centric subtasks，然后根据新环境中的物体位置对轨迹片段进行变换、重放和组合，只保留成功执行的轨迹。

- 预训练仿真任务主要在 RoboCasa 框架下构建。
- 任务形式通常是 `rearrange A from B to C`。
- 包括 54 种 source / target receptacle category 组合。
- 每种组合生成约 10,000 条示教，预训练阶段总计约 540k 条仿真示教。
- 如果考虑 pre-training 与 post-training，论文称仿真生成约 780,000 条 trajectories，相当于约 6,500 小时的人类示教数据，生成只需约 11 小时。

**E. 数据规模**

论文统计的预训练数据总量约为：

- **总数据**：592.9M frames，8,375.7 hours；
- **robot data**：262.3M frames，3,288.8 hours；
- **human data**：181.3M frames，2,517.0 hours；
- **simulation data**：125.5M frames，1,742.6 hours；
- **neural data**：23.8M frames，827.3 hours。

> 这说明 GR00T N1 的核心贡献不只是模型架构，也包括把真实机器人、仿真、人类视频和神经生成视频统一到同一个 VLA 训练框架中的数据工程方案。

---

### 五. 实验

**实验目标**：论文主要验证 GR00T N1 是否能通过大规模预训练获得更强的泛化能力，并在少量下游数据上比从零训练的 imitation learning baseline 更高效地适应新任务。

**A. Simulation Benchmarks**

仿真实验包括三个 benchmark：

- **RoboCasa Kitchen**
  - 24 个 atomic kitchen manipulation tasks。
  - 使用 Franka Emika Panda。
  - 包括 pick-and-place、开关门、按按钮、开关水龙头、开关微波炉等任务。
  - 每个任务使用 3000 条公开示教数据。

- **DexMimicGen Cross-Embodiment Suite**
  - 9 个双臂 / 灵巧手任务。
  - 覆盖三类 embodiment：双 Panda 夹爪、双 Panda 灵巧手、GR-1 humanoid dexterous hands。
  - 包括 threading、piece assembly、transport、box cleanup、drawer cleanup、tray lifting、pouring、coffee preparation、can sorting 等任务。

- **GR-1 Tabletop Tasks**
  - 24 个 humanoid tabletop manipulation tasks。
  - 使用 GR-1 humanoid 和 Fourier dexterous hands。
  - 包括 18 个 rearrangement tasks，以及 6 个涉及 articulated objects 的任务，例如放入 cabinet、drawer、microwave 并关闭。
  - 更接近真实 humanoid tabletop deployment。

**B. Real-World Benchmarks**

真实机器人实验在 Fourier GR-1 humanoid 上进行，任务分为四类：

- **Object-to-Container Pick-and-Place**
  - 5 个任务。
  - 测试抓取、放置、空间对齐和对新物体的泛化。

- **Articulated Object Manipulation**
  - 3 个任务。
  - 需要把物体放入 white drawer、dark cabinet、wooden chest 等带关节结构的容器，并完成关闭。

- **Industrial Object Manipulation**
  - 3 个任务。
  - 包括 machinery packing、mesh cup pouring、cylinder handover。
  - 更接近结构化工业场景。

- **Multi-Agent Coordination**
  - 2 个任务。
  - 涉及两个机器人之间的协作，例如把 cylinder 放入 mesh cup 并交给另一台机器人。

**C. 预训练模型的开箱即用能力**

论文在真实 GR-1 humanoid 上设计了两个预训练评估任务：

- **双手协调任务**：物体位于左手左侧，机器人需要先用左手抓取，再 transfer 到右手可达范围内，最后放到底层架子上。
- **新物体新容器任务**：要求机器人把 novel object 放入 unseen target container。

GR00T-N1-2B 在这两个任务上的成功率分别为：

- 双手协调任务：**76.6%**；
- novel object / unseen container：**73.3%**。

> 这说明大规模预训练确实赋予了模型一定的 zero-shot / out-of-box 行为能力，尤其是在双手协作和新物体泛化方面。

**D. Post-training 仿真结果**

在仿真 benchmark 中，论文比较了 GR00T-N1-2B 与 BC Transformer、Diffusion Policy。使用每任务 100 条 demonstration 时，平均成功率如下：

| 模型 | RoboCasa | DexMG | GR-1 | Average |
|---|---:|---:|---:|---:|
| BC Transformer | 26.3% | 53.9% | 16.1% | 26.4% |
| Diffusion Policy | 25.6% | 56.1% | 32.7% | 33.4% |
| GR00T-N1-2B | 32.1% | 66.5% | 50.0% | 45.0% |

**核心结论**：GR00T N1 在三个仿真 benchmark 上均超过 baseline，尤其在 GR-1 humanoid tabletop tasks 上优势明显，相比 Diffusion Policy 高出约 **17.3%**。

**E. Post-training 真实机器人结果**

在真实 GR-1 humanoid 上，论文比较了 Diffusion Policy 和 GR00T-N1-2B，并分别测试 10% 数据和完整数据。

| 模型 | Pick-and-Place | Articulated | Industrial | Coordination | Average |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy (10% Data) | 3.0% | 14.3% | 6.7% | 27.5% | 10.2% |
| Diffusion Policy (Full Data) | 36.0% | 38.6% | 61.0% | 62.5% | 46.4% |
| GR00T-N1-2B (10% Data) | 35.0% | 62.0% | 31.0% | 50.0% | 42.6% |
| GR00T-N1-2B (Full Data) | 82.0% | 70.9% | 70.0% | 82.5% | 76.8% |

**核心结论**：GR00T-N1-2B 在低数据和全数据设置下都显著优于 Diffusion Policy。尤其值得注意的是，GR00T-N1-2B 只用 **10% 数据** 就达到 **42.6%** 平均成功率，已经接近 Diffusion Policy 使用完整数据的 **46.4%**，说明预训练显著提高了数据效率。

**F. Neural Trajectories 消融实验**

论文进一步研究在 post-training 中加入 neural trajectories 的效果。结果显示，与只使用真实轨迹相比，加入 neural trajectories 后：

- RoboCasa 在 30 / 100 / 300 demos per task 三种数据规模下分别提升约 **+4.2% / +8.8% / +6.8%**；
- 真实 GR-1 humanoid 的 8 个任务平均提升约 **+5.8%**。

> 这说明神经生成视频不仅可以作为预训练数据，也可以在低数据 post-training 阶段作为一种有效的数据增强方式。

---

### 六. 局限性

* **短视距 tabletop manipulation 为主**：GR00T N1 当前主要聚焦于 short-horizon tabletop manipulation。虽然任务已经包括双手操作、工业物体操作和 multi-agent coordination，但距离 long-horizon loco-manipulation 还有明显差距。未来若要处理移动、导航、全身控制和长程任务组合，需要更强的 humanoid 硬件、模型架构和训练数据。

* **空间推理和语言理解仍依赖更强 VLM**：论文认为更强的 vision-language backbone 有望提升空间推理、语言理解和任务适应能力。这意味着当前 System 2 的语义理解仍可能成为复杂场景下的性能瓶颈，尤其是在细粒度空间关系、遮挡、多物体歧义和长程指令理解方面。

* **合成数据质量仍受限制**：GR00T N1 的数据金字塔高度依赖 simulation trajectories 和 neural trajectories，但现有合成数据方法仍难以同时满足多样性、反事实生成能力和物理一致性。video generation model 可以生成丰富轨迹，但可能违反物理规律；simulation 可以保证物理一致性，但存在 sim-to-real gap，且对液体、柔性物体、复杂接触和 articulated objects 的覆盖仍有限。

* **泛化能力仍需更系统验证**：论文证明了 GR00T N1 在多种 manipulation benchmark 和 GR-1 humanoid 上具有较强效果，但它的“generalist humanoid foundation model”能力尚未覆盖更广泛的真实人类环境任务，例如长时间家庭任务、移动操作、动态人机协作和跨硬件平台大规模部署。因此，当前结论更准确地说是：GR00T N1 在 humanoid tabletop manipulation 上展示了基础模型路线的可行性，而不是已经解决通用 humanoid autonomy。

---

### 七. 与 π0 的关系

GR00T N1 与 $\pi_0$ 都属于 **VLM + continuous action generation** 路线，但二者的架构侧重点不同。

- $\pi_0$ 的核心是 **预训练 VLM + Action Expert + Flow Matching**。它在单一 Transformer 框架中引入独立动作专家，让视觉语言 token 与动作 token 在统一自注意力机制中交互。
- GR00T N1 的核心是 **Dual-System VLA**。它将 VLM 明确作为 System 2，将 DiT action module 明确作为 System 1，并通过 cross-attention 将视觉语言 token 注入动作生成。
- $\pi_0$ 更强调通用机器人控制中的高频连续动作生成和灵巧操作能力；GR00T N1 更强调 humanoid embodiment、跨 embodiment 数据混合、以及 real / simulation / neural / human video 的数据金字塔。
- 两者都使用 flow matching 和 action chunking，但 GR00T N1 的动作块长度为 $H=16$，推理中通常使用 $K=4$ 步去噪；$\pi_0$ 则使用更长的动作块（$H=50$）并强调低频规划（2 Hz）、高频执行（50Hz）。

> 简单概括：$\pi_0$ 是“VLM + Action Expert”的通用机器人控制模型，GR00T N1 是“System 2 VLM + System 1 DiT”的 humanoid robot foundation model。前者更像把动作生成嵌入 VLM 框架，后者更像把 VLM 语义理解和 DiT 动作生成组合成一个快慢系统。
