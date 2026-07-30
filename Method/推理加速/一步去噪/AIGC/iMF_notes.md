## Improved Mean Flows: On the Challenges of Fastforward Generative Models

> 论文：https://arxiv.org/pdf/2512.02012 
> 代码：https://github.com/Lyy-iiis/imeanflow

### 一. 概述

**研究动机**：普通 Flow Matching 学的是**某个时间点附近的局部速度**。它适合多步 ODE integration，但如果只用 1-NFE，即从噪声一步生成数据，模型很难直接完成大跨度跳跃。MeanFlow 因此提出让模型学习**从当前时间到目标时间这一整段的平均速度**，从而支持一步/少步生成。

不过原始 MeanFlow 有两个主要问题：

1. **训练目标依赖网络自身，不是标准回归问题。** 
   原始 MeanFlow 的真实平均速度不可直接获得，于是用网络自己的预测参与构造训练目标。这会导致 target 不是一个固定的、网络无关的监督信号，训练不够稳定。

2. **CFG scale 在训练时固定，推理时不灵活。** 
   原始 MeanFlow 为了支持 1-NFE classifier-free guidance，需要在训练时固定 guidance scale。但不同模型大小、训练阶段、推理步数下，最优 guidance scale 可能不同。固定它会限制推理时调参空间。

   > CFG（Classifier-Free Guidance）可以理解为“让生成结果更听条件的话”。推理时，模型通常会做两次预测：
   >
   > 1. **conditional prediction**：带条件预测，例如在类别 / 文本条件 $c$ 下预测：

$$
v_{cond}=v_\theta(z_t,t|c)
$$

   >
   > 2. **unconditional prediction**：不带条件预测，即去掉条件后的预测：

$$
v_{uncond}=v_\theta(z_t,t|\emptyset)
$$

   >
   > 然后用二者差值表示“条件带来的修正方向”：
   >

$$
v_{cond}-v_{uncond}
$$

   >
   > 最终预测为：
   >

$$
v_{cfg} = v_{uncond} + s\cdot (v_{cond}-v_{uncond})
$$

   >
   > 其中 $s$ 是 guidance scale。 $s$ 越大，模型越强调条件约束；但过大可能降低多样性，甚至导致生成失真。
   >
   > > CFG 和 OFP 的 Self-Guided Regularization 都利用了 conditional / unconditional 预测的差值，表示“条件带来的修正方向”。区别在于：
   > >
   > > * CFG：在推理阶段使用。每次生成都需要 conditional / unconditional 两次预测，再组合得到 guided prediction。
   > >   * 训练时先让同一个模型具备两种预测能力，真正的 guidance 组合通常在推理时做。
   > > * OFP Self-Guided Regularization：在训练阶段使用，用 conditional / unconditional 的差值构造 pseudo target，让 student 学会这种修正方向，推理时不需要额外做两次预测。

   因此 iMF 的目标是：**保留 MeanFlow 的“平均速度“和”一步生成“思想，但把训练目标改得更稳定，并让 guidance 在推理时可调。**

**核心思想**：模型仍然预测**平均速度**，但训练时把目标重新表述成一个更标准的**局部速度形式**。

原始 MeanFlow 可以理解为：

> 模型想直接预测平均速度 $u_\theta$ ，也就是从当前时间 $t$ 到目标时间 $r$ 这一整段的平均速度。但问题是，真实的平均速度不能像普通 Flow Matching 里的 $e-x$ 那样直接写出来。所以原始 MeanFlow 需要借助 MeanFlow identity，并**让模型当前自己的预测 $u_\theta$ 参与构造一个伪 target**。这样训练目标不是一个完全固定的监督信号，而是会随着模型自身变化，因此训练会更复杂，也更不稳定。

iMF 的做法可以理解为：

> 模型仍然输出 average velocity $u_\theta$ ，但**训练时不直接拿 $u_\theta$ 去和某个平均速度 target 做比较**。相反，它先利用 MeanFlow identity，**把 $u_\theta$ 转换成一个“等价的瞬时速度预测” $V_\theta$**，然后让这个 $V_\theta$ 去拟合普通 Flow Matching 中已知的速度目标 $e-x$ 。这样做的好处是：模型最终学到的仍然是平均速度，但 loss 计算时使用的是更稳定、更容易监督的 velocity target。

简单概括：

* MeanFlow：直接训练 average velocity，目标构造复杂。
* Improved MeanFlow：网络仍预测 average velocity，但通过 MeanFlow identity 转换成 $V_θ$ ，再用标准 velocity target 监督。

---

### 二. 模型方法

iMF 主要包含三个改进：**更稳定的 v-loss 形式、灵活的 guidance conditioning、in-context conditioning 架构。**

#### 1. 从 u-loss 改写为 v-loss

普通 Flow Matching 使用线性路径：

$$
z_t=(1-t)x+te
$$

其中 $x$ 是真实数据， $e$ 是噪声。对应的 conditional velocity 是：

$$
v_c=e-x
$$

原始 MeanFlow 定义从 $r$ 到 $t$ 的 average velocity：

$$
u(z_t,r,t) = \frac{1}{t-r}\int_r^t v(z_\tau, \tau)d\tau
$$

iMF 通过 MeanFlow identity 建立 instantaneous velocity 与 average velocity 的关系：

$$
v(z_t,t)=u(z_t,r,t)+(t−r)\frac{d}{dt}u(z_t,r,t)
$$

一句话理解：**局部速度 = 当前这段平均速度 + 平均速度随时间变化带来的修正项**

<img src="./images/MeanFlow identity.png" alt="MeanFlow identity" style="zoom:33%;" />

> 这个公式来自 average velocity 的定义：

$$
u(z_t,r,t) = \frac{1}{t-r} \int_r^t v(z_\tau,\tau)d\tau
$$

> 两边乘上 $(t-r)$ ：

$$
(t-r)u(z_t,r,t) = \int_r^t v(z_\tau,\tau)d\tau
$$

> 右边表示：从 $r$ 到 $t$ 这段时间里，沿着生成轨迹累计走过的总位移。
>
> 现在对 $t$ 求导。根据“积分上限求导”：

$$
\frac{d}{dt} \int_r^t v(z_\tau,\tau)d\tau = v(z_t,t)
$$

> 左边对 $(t-r)u(z_t,r,t)$ 求导，用乘法法则：

$$
\frac{d}{dt} \left[ (t-r)u(z_t,r,t) \right] = u(z_t,r,t) + (t-r)\frac{d}{dt}u(z_t,r,t)
$$

> 所以得到：

$$
v(z_t,t) = u(z_t,r,t) + (t-r)\frac{d}{dt}u(z_t,r,t)
$$

iMF 用网络 $u_\theta$ 预测 average velocity，并通过 JVP 近似 $\frac{d}{dt}u_\theta$ ，得到复合函数：

$$
V_\theta(z_t) = u_\theta(z_t,r,t) + (t-r)\mathrm{JVP}_{sg}(u_\theta; v_\theta)
$$

> JVP 就是 iMF 用来计算 $\frac{d}{dt}u_\theta$ 的自动微分操作：

$$
\mathrm{JVP}(u_θ;v_θ)=\frac{∂u_θ}{∂z_t}v_θ+\frac{∂u_θ}{∂t}
$$

> 它表示 average velocity 随当前时间和当前位置变化的趋势。

然后用 $V_\theta$ 去拟合标准 Flow Matching 的速度目标：

$$
L_{iMF} = \left\| V_\theta(z_t)-(e-x) \right\|^2
$$

直观理解是：**训练时不再直接监督未知的 average velocity，而是让 average velocity 经过 MeanFlow identity 后，能够还原普通 Flow Matching 的 conditional velocity。**

#### 2. 用 $v_\theta$ 替代 $e-x$ 作为 JVP 方向

原始 MeanFlow 在计算 JVP 时，直接使用当前样本对 $(x,e)$ 的直线速度 $e-x$ 作为瞬时速度方向。也就是问：如果 $z_t$ 沿着这条样本直线的方向移动， $u_\theta$ 会怎么变化？

但问题是， $e-x$ 是某个具体样本对的 conditional velocity。同一个 $z_t$ 可能对应很多不同的 $(x,e)$ 组合，因此**不同样本给出的 $e-x$ 方差很大，会让 JVP 项不稳定**。

iMF 的做法是：不用某个样本对的 $e-x$ 作为 JVP 方向，而是用**模型预测的局部速度 $v_\theta(z_t,t)$**：

$$
V_\theta(z_t) = u_\theta(z_t,r,t) + (t-r)\mathrm{JVP}_{sg}(u_\theta;v_\theta)
$$

这样 JVP 的方向**只依赖当前 noisy state $z_t$**，更接近模型推理时真正使用的速度场，也更稳定。

> $v_\theta(z_t,t)$ 可以通过 boundary condition 得到：

$$
v_\theta(z_t,t)=u_\theta(z_t,t,t)
$$

>
> 也就是说，当目标时间 $r$ 等于当前时间 $t$ 时，区间长度趋近于 0，average velocity 就退化成 instantaneous velocity。

#### 3. Flexible Guidance Conditioning

原始 MeanFlow 为了支持 CFG，需要在训练时固定 guidance scale $\omega$ 。iMF 认为这不合理，因为不同设置下最优 $\omega$ 会变化。比如模型更强、训练更久、或者推理步数更多时，最优 CFG scale 可能会变小。

iMF 因此**把 guidance scale $\omega$ 当成一个显式条件输入给网络**：

$$
u_\theta = u_\theta(z_t \mid r,t,c,\omega)
$$

这样**训练时可以随机采样不同的 $\omega$ ，推理时也可以自由调整 $\omega$**。

进一步地，iMF 还支持 CFG interval，也就是只在某个时间区间 $[t_{min},t_{max}]$ 内启用 CFG。于是 guidance-related conditions 可以写成：

$$
\Omega=\{\omega,t_{min},t_{max}\}
$$

完整网络条件为：

$$
u_\theta = u_\theta(z_t \mid r,t,c,\Omega)
$$

#### 4. Improved In-context Conditioning

<img src="./images/in-context conditioning.png" alt="in-context conditioning" style="zoom: 33%;" />

iMF 中条件包括很多种类：

* time steps： $r, t$
* class condition： $c$
* guidance condition： $Ω = \{ω, t_{min}, t_{max}\}$

传统 DiT 常用 adaLN-zero 注入条件信息：**先把时间、类别等条件编码成 embedding，再由这些 embedding 生成 scale / shift / gate 等调节参数，用来调节每个 Transformer block 中的特征**。iMF 认为，当条件类型很多时，把所有条件压成这种调节信号不够灵活，而且每层都需要额外的调节参数，参数量较大。因此，iMF 改用 in-context conditioning，**把不同条件分别变成 condition tokens，和 image tokens 拼接后一起送入 Transformer**。

这样做有两个好处：

1. 可以灵活处理多种异质条件；
2. 可以移除参数量较大的 adaLN-zero，使模型参数减少约 1/3。

---

### 三. 训练流程

iMF 的训练流程可以简化为下面几步。

#### 1. 采样数据和时间

输入真实数据 $x$ ，采样噪声 $e\sim \mathcal{N}(0,I)$ ，采样时间 $t,r \sim \mathrm{LogitNormal}(\mu=-0.4,\sigma=1.0)$ ，加噪得到当前时间的带噪数据 $z_t$ ：

$$
z_t=(1-t)x+te
$$

其中 $t$ 表示当前噪声时间， $r$ 表示目标时间；若 $r=0,t=1$ ，对应从纯噪声一步生成数据。

并以 0.5 的概率设置 $r=t$ ，从而让一半样本退化为普通 Flow Matching。

#### 2. 预测 instantaneous velocity

利用 boundary condition，让网络在 $r=t$ 的情况下预测 instantaneous velocity：

$$
v_\theta = u_\theta(z_t,t,t)
$$

这一步得到的 $v_\theta$ 用作 JVP 的切向量，而不是直接使用 $e-x$ 。

#### 3. 预测 average velocity 和 JVP

让网络预测从 $t$ 到 $r$ 的 average velocity：

$$
u_\theta = u_\theta(z_t,r,t)
$$

同时通过 JVP 计算：

$$
\mathrm{JVP}(u_\theta; v_\theta)
$$

其中 JVP 可以理解为：沿着 $v_\theta$ 这个方向，average velocity $u_\theta$ 会如何变化。

#### 4. 构造 compound prediction

根据 MeanFlow identity 构造：

$$
V_\theta(z_t) = u_\theta(z_t,r,t) + (t-r)\mathrm{JVP}_{sg}(u_\theta;v_\theta)
$$

这里的 sg 表示使用 stop-gradient，避免高阶梯度导致优化困难。

#### 5. 计算训练损失

用普通 Flow Matching 的 conditional velocity 作为监督：

$$
v_c=e-x
$$

计算：

$$
L_{iMF} = \left\| V_\theta - (e-x) \right\|^2
$$

> 如果使用 guidance conditioning，iMF 会在训练时随机采样 guidance scale $\omega$ ，并把 $\omega$ 作为输入条件。模型用同一套参数分别计算 conditional（ $fn(z, t, t, ω, c)$ ） 和 unconditional（ $fn(z, t, t, ω)$ ）预测，用二者差异构造 guidance-aware target。训练的目的不是分别优化两个模型，而是让同一个模型学会：给定条件 $c$ 和 guidance scale $\omega$ ，直接输出对应 guidance 强度下的 average velocity。因此推理时只需输入 $c,\omega$ ，不需要再额外做 conditional / unconditional 两次预测。

---

### 四. 推理流程

iMF 推理时非常简单，因为网络已经学会了 average velocity。对于 one-step generation：

1. 从噪声分布采样：

$$
z_1\sim \mathcal{N}(0,I)
$$

2. 设置目标区间：

$$
r=0,\quad t=1
$$

3. 网络预测从 $t=1$ 到 $r=0$ 的 average velocity：

$$
u_\theta(z_1,0,1)
$$

4. 一步生成数据：

$$
z_0=z_1-u_\theta(z_1,0,1)
$$

> 这里符号方向与论文中的时间约定有关：论文采用 $z_t=(1-t)x+te$ ，所以 $t=1$ 是噪声， $t=0$ 是数据。从 $t=1$ 走到 $r=0$ ，就是从噪声生成数据。

如果使用 few-step generation，也可以把区间拆成几段，每段用 average velocity 跳一次。例如 2-NFE 可以理解为：

```text
z_1 → z_0.5 → z_0
```

但 iMF 的重点是：**即使只用 1-NFE，也能直接生成高质量样本。**

---

### 五. 对 Fast-WAM 的启发

iMF 对 Fast-WAM 最重要的启发是：**如果 2-step 能保持性能但 1-step 明显下降，说明模型可能只学好了局部速度 / 小步修正，没有被专门训练去完成大跨度的一步跳跃。**

iMF 最值得迁移的思路是：**用 iMF-style objective 训练 1-step action expert**

> 把图像生成中的数据 $x$ 换成 Fast-WAM 的 action chunk $a$ ，把噪声 $e$ 换成 action noise $\epsilon$ ：
>

$$
z_t=(1-t)a+t\epsilon
$$

>
> 让 action expert 输出 average velocity：
>

$$
u_\theta(z_t,r,t \mid \text{world context})
$$

>
> 再构造：
>

$$
V_\theta = u_\theta + (t-r)\mathrm{JVP}_{sg}(u_\theta;v_\theta)
$$

>
> 并用：
>

$$
\epsilon-a
$$

>
> 作为 velocity target：
>

$$
L = \left\| V_\theta-(\epsilon-a) \right\|^2
$$

>
> 这样 action expert 不只是学习局部 velocity，而是学习在给定 world context 下，从 noisy action 一步跳到 clean action 所需的区间平均速度。

iMF 的方法来自图像生成，直接迁移到 Fast-WAM 时可能会出现 **JVP 训练成本可能较高**的问题，因为 Fast-WAM 本身模型较大。

---

### 六. 总结

iMF 可以理解为：**保留 MeanFlow 的 average velocity 思想，但把训练目标改写成更稳定的 v-loss，并通过 flexible conditioning 支持更灵活的 one-step generation。**

对 Fast-WAM 来说，它最值得借鉴的是：**网络输出 average velocity，但 loss 可以放在更稳定的 velocity space**。
