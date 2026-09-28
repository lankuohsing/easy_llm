# DPO（Direct Preference Optimization）基础理论

## 摘要

Direct Preference Optimization（DPO，直接偏好优化）是一类利用成对偏好数据训练语言模型的方法 [1]。经典 RLHF 通常先从偏好数据拟合奖励模型，再通过 PPO 等强化学习算法优化策略 [2,3]；DPO 则利用 KL 正则化奖励最大化问题的闭式最优策略，将潜在奖励直接改写为策略相对参考策略的对数概率比，从而把策略优化归约为一个二元分类问题。训练时，DPO 只需要偏好样本、待训练策略和冻结的参考策略，不需要显式奖励模型、价值模型或在线 rollout。

本文从问题设定出发，依次推导 Bradley–Terry 偏好模型、KL 正则化最优策略和 DPO 损失，分析其梯度、参考策略与超参数的作用，并讨论序列概率计算、数据条件、评估方法及理论边界。本文所称的“DPO”特指 Rafailov 等人提出的基础形式；工程框架中的变体可能采用不同的损失、归一化或正则化方式。

**关键词：** 大语言模型；偏好学习；RLHF；KL 正则化；直接偏好优化；Bradley–Terry 模型

## 1. 引言

监督微调（Supervised Fine-Tuning，SFT）通过最大化参考回答的条件似然，使语言模型学会遵循指令。然而，开放式生成往往不存在唯一标准答案。对于同一提示词，人们可能同时接受多个回答，却能较稳定地判断“回答 A 比回答 B 更好”。成对偏好数据因此成为描述帮助性、安全性、风格和任务质量的重要监督形式 [4]。

经典的基于人类反馈的强化学习（Reinforcement Learning from Human Feedback，RLHF）通常包含三个阶段：

```text
高质量示范 ──SFT──> 初始策略
偏好对 ──奖励建模──> 奖励模型
初始策略 + 奖励模型 ──PPO──> 对齐后的策略
```

这条路线能够直接优化由奖励模型给出的序列级信号，但需要维护策略模型、参考模型、奖励模型和价值模型，并在训练期间反复生成回答。DPO 的出发点是：如果奖励最优策略与奖励函数之间存在可逆关系，就可以不显式恢复奖励函数，而直接用偏好数据拟合最优策略。

DPO 的训练流程可概括为：

```text
同一提示词下的偏好回答对
  ↓
当前策略与参考策略分别计算 chosen / rejected 的序列 log probability
  ↓
构造两条回答的相对对数概率比差值
  ↓
二元交叉熵损失
  ↓
只更新当前策略
```

这种训练形式与普通微调相似，但其目标并不是复现一个答案，而是让当前策略相对参考策略更偏向获选回答。因此，DPO 既不同于只使用 chosen 回答的 SFT，也不同于依赖在线采样和显式奖励的 PPO。

## 2. 问题定义与符号约定

### 2.1 成对偏好数据

设偏好数据集为

$$
\mathcal{D}=\{(x_i,y_i^{w},y_i^{l})\}_{i=1}^{N}
$$

其中，同一提示词 $x_i$ 对应两个候选回答：标注者更偏好的回答记为 $y_i^{w}$ ，较不偏好的回答记为 $y_i^{l}$ 。上标 $w$ 和 $l$ 分别来自 winner 与 loser。偏好关系写作 $y_i^{w}\succ y_i^{l}$ 。

本文使用以下符号：

| 符号 | 含义 |
| --- | --- |
| $\mathcal{D}$ | 由成对偏好样本组成的训练集 |
| $N$ | 训练集中的偏好对数量 |
| $\rho(x)$ | KL 正则化奖励目标中的提示词分布 |
| $x$ | 提示词或对话上下文，可包含 system 消息、历史消息和当前用户指令 |
| $y^{w}$ | 同一提示词下获选的回答，即 chosen / preferred response |
| $y^{l}$ | 同一提示词下未获选的回答，即 rejected / dispreferred response |
| $\pi_\theta(y\mid x)$ | 参数为 $\theta$ 的当前策略对完整回答 $y$ 给出的条件概率 |
| $\pi_{\mathrm{ref}}(y\mid x)$ | 冻结的参考策略对完整回答 $y$ 给出的条件概率 |
| $r(x,y)$ | 对提示词—回答对进行评价的潜在标量奖励 |
| $\beta$ | KL 正则化系数，满足 $\beta\gt0$ |
| $Z(x)$ | 只依赖提示词 $x$ 的配分函数 |
| $\sigma(z)$ | Sigmoid 函数， $\sigma(z)=1/(1+\exp(-z))$ |
| $m_\theta(x,y^{w},y^{l})$ | 当前策略相对参考策略的偏好间隔 |
| $\mathcal{L}_{\mathrm{DPO}}(\theta)$ | 需要最小化的 DPO 经验损失 |

这里沿用强化学习中的术语，将自回归语言模型称为策略（policy）。与 PPO 不同，基础 DPO 训练中没有用于生成当前批次数据的旧策略 $\pi_{\mathrm{old}}$ ，也没有价值模型。数据集中的回答通常在训练开始前已经收集完毕。

### 2.2 完整回答的概率

设回答 $y=(y_1,\ldots,y_T)$ 包含 $T$ 个有效 token。自回归策略对完整回答的条件概率为

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{\lt t})
$$

计算时通常使用对数概率之和：

$$
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{\lt t})
$$

prompt token 只提供条件，不计入回答的序列对数概率；回答内容和结束 token 通常计入，padding 必须屏蔽。根据自回归链式法则，完整回答的序列对数概率等于各回答 token 的对数概率之和，基础 DPO 直接使用这一序列对数概率。某些实现或变体会再除以回答 token 数量，以得到平均每个 token 的对数概率并减弱长度的影响；这种长度归一化会改变目标函数，不属于基础 DPO 的定义。

## 3. 从偏好到潜在奖励

### 3.1 Bradley–Terry 偏好模型

成对偏好学习需要描述“回答 $y_1$ 胜过回答 $y_2$ 的概率”。DPO 采用 Bradley–Terry 模型 [5]：假设每个回答具有潜在奖励 $r(x,y)$ ，偏好概率由奖励差决定：

$$
p^*(y_1\succ y_2\mid x)=\frac{\exp(r(x,y_1))}{\exp(r(x,y_1))+\exp(r(x,y_2))}=\sigma\left(r(x,y_1)-r(x,y_2)\right)
$$

该模型表达了三个基本性质：

1. 偏好只取决于同一提示词下两个回答的奖励差；
2. 奖励差为 0 时，两个回答各有 $0.5$ 的获选概率；
3. 奖励差越大，高奖励回答被选择的概率越接近 1。

因为偏好只依赖奖励差，所以对同一提示词的所有奖励同时加上任意常数 $c(x)$ ，偏好概率保持不变。换言之，成对比较无法识别奖励的绝对零点，只能识别相对差异。

### 3.2 经典奖励模型目标

若显式训练参数为 $\phi$ 的奖励模型 $r_\phi$ ，可对观测到的偏好标签做最大似然估计：

$$
\mathcal{L}_{\mathrm{RM}}(\phi)=-\mathbb{E}_{(x,y^{w},y^{l})\sim\mathcal{D}}\left[\log\sigma\left(r_\phi(x,y^{w})-r_\phi(x,y^{l})\right)\right]
$$

经典 RLHF 随后把 $r_\phi$ 当作奖励函数，通过强化学习优化语言模型。DPO 则不单独训练这样的奖励模型，而是利用奖励函数与 KL 正则化最优策略之间的解析关系，将奖励建模吸收到策略参数化和偏好目标中。直观地说，奖励不再由一个独立模型显式输出，而是由当前策略相对参考策略的概率变化隐式表示；DPO 再用这一隐式奖励构造偏好概率并直接优化 $\pi_\theta$ 。第 5.3 节将给出隐式奖励的具体形式。

## 4. KL 正则化奖励最大化

### 4.1 优化目标

给定提示词分布 $x\sim\rho$ 、参考策略 $\pi_{\mathrm{ref}}$ 和奖励函数 $r$ ，常见的 RLHF 策略目标为

$$
\pi_r^*=\underset{\pi}{\arg\max}\ \left\{\mathbb{E}_{x\sim\rho,\,y\sim\pi(\cdot\mid x)}[r(x,y)]-\beta\mathbb{E}_{x\sim\rho}\left[D_{\mathrm{KL}}\left(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)\right]\right\}
$$

第一项鼓励策略生成高奖励回答；第二项惩罚策略偏离参考策略。这里使用正向 KL 散度

$$
D_{\mathrm{KL}}\left(\pi(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)\right)=\mathbb{E}_{y\sim\pi(\cdot\mid x)}\left[\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\right]
$$

系数 $\beta$ 决定奖励提升与行为保持之间的权衡。理论上，较大的 $\beta$ 对偏离参考策略施加更强惩罚；较小的 $\beta$ 允许最优策略为追求奖励而发生更大变化。

### 4.2 命题：固定提示词时的闭式最优策略

对一个固定提示词 $x$ ，考虑第 4.1 节定义的 KL 正则化奖励最大化问题。为保证其中的 KL 散度有限，假设待优化策略不会为参考策略概率严格为 0 的回答分配非零概率；普通 softmax 语言模型通常自动满足这一条件。进一步假设 $\beta\gt0$ 且下文定义的 $Z(x)$ 有限。策略还需要满足概率归一化约束，相应目标写为

$$
J_x(\pi)=\sum_y\pi(y\mid x)r(x,y)-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

为约束 $\sum_y\pi(y\mid x)=1$ 引入拉格朗日乘子。对每个 $\pi(y\mid x)$ 求驻点，可以得到

$$
r(x,y)-\beta\left(\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+1\right)+\lambda(x)=0
$$

整理并归一化后，最优策略为

$$
\pi_r^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

其中，配分函数为

$$
Z(x)=\sum_y\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

该闭式解表明，最优策略是在参考策略上按指数化奖励重新加权：奖励更高的回答获得更多概率，但参考策略概率很低的回答仍受到抑制。KL 正则化因而不仅是训练稳定技巧，也是决定目标策略分布的组成部分。

上述命题的完整拉格朗日推导见附录 A。

### 4.3 从最优策略反解奖励

对闭式解取对数并移项，可将奖励写成

$$
r(x,y)=\beta\log\frac{\pi_r^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)
$$

第一项衡量最优策略相对参考策略提升了多少回答概率；第二项只依赖提示词，不依赖回答。这个等式构成 DPO 的关键桥梁：**策略相对参考策略的对数概率比，可以充当隐式奖励。**

## 5. DPO 目标函数的推导

### 5.1 配分函数为何消失

把反解出的奖励代入 Bradley–Terry 模型。对于同一提示词 $x$ 下的两个回答，有

$$
\begin{aligned} r(x,y^{w})-r(x,y^{l}) &=\beta\log\frac{\pi_r^*(y^{w}\mid x)}{\pi_{\mathrm{ref}}(y^{w}\mid x)}-\beta\log\frac{\pi_r^*(y^{l}\mid x)}{\pi_{\mathrm{ref}}(y^{l}\mid x)} \\ &\quad+\beta\log Z(x)-\beta\log Z(x) \\ &=\beta\log\frac{\pi_r^*(y^{w}\mid x)}{\pi_{\mathrm{ref}}(y^{w}\mid x)}-\beta\log\frac{\pi_r^*(y^{l}\mid x)}{\pi_{\mathrm{ref}}(y^{l}\mid x)} \end{aligned}
$$

不可计算的 $Z(x)$ 在奖励差中精确抵消。令可训练策略 $\pi_\theta$ 近似最优策略，DPO 对偏好标签建立如下概率模型：

$$
p_\theta(y^{w}\succ y^{l}\mid x)=\sigma\left(\beta\log\frac{\pi_\theta(y^{w}\mid x)}{\pi_{\mathrm{ref}}(y^{w}\mid x)}-\beta\log\frac{\pi_\theta(y^{l}\mid x)}{\pi_{\mathrm{ref}}(y^{l}\mid x)}\right)
$$

因此，DPO 不需要枚举所有可能回答来计算 $Z(x)$ ，也不需要先得到一个数值奖励模型。

### 5.2 偏好间隔与经验损失

定义策略相对参考策略的偏好间隔

$$
m_\theta(x,y^{w},y^{l})=\log\frac{\pi_\theta(y^{w}\mid x)}{\pi_{\mathrm{ref}}(y^{w}\mid x)}-\log\frac{\pi_\theta(y^{l}\mid x)}{\pi_{\mathrm{ref}}(y^{l}\mid x)}
$$

对数据集中“$y^{w}$ 胜过 $y^{l}$”这一观测做负对数似然估计，得到基础 DPO 损失：

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)=-\mathbb{E}_{(x,y^{w},y^{l})\sim\mathcal{D}}\left[\log\sigma\left(\beta m_\theta(x,y^{w},y^{l})\right)\right]
$$

有限数据集上的经验平均为

$$
\mathcal{L}_{\mathrm{DPO}}(\theta)=-\frac{1}{N}\sum_{i=1}^{N}\log\sigma\left(\beta m_\theta(x_i,y_i^{w},y_i^{l})\right)
$$

最优参数写作

$$
\theta^*=\underset{\theta}{\arg\min}\ \mathcal{L}_{\mathrm{DPO}}(\theta)
$$

从统计学习视角看，DPO 是以 $\beta m_\theta$ 为 logit 的二元逻辑回归；从 RLHF 视角看，它通过偏好似然拟合 KL 正则化奖励目标所对应的最优策略。这两个解释描述的是同一个目标。

### 5.3 隐式奖励

DPO 策略对应的隐式奖励可写为

$$
\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
$$

它与理论奖励相差一个只依赖 $x$ 的常数 $\beta\log Z(x)$ 。因为成对偏好只依赖奖励差，这个常数不会影响训练。需要注意， $\hat r_\theta$ 不是跨提示词可直接比较的绝对评分：不同提示词对应的未知平移量可能不同。

## 6. DPO 的优化机制

### 6.1 单个偏好对的梯度

对单个样本记 $z_\theta=\beta m_\theta(x,y^{w},y^{l})$ ，其损失为 $\ell(\theta)=-\log\sigma(z_\theta)$ 。参考策略保持冻结，因此梯度为

$$
\nabla_\theta\ell(\theta)=-\beta\sigma(-z_\theta)\left[\nabla_\theta\log\pi_\theta(y^{w}\mid x)-\nabla_\theta\log\pi_\theta(y^{l}\mid x)\right]
$$

梯度下降由此产生两种相反作用：

- 提高获选回答 $y^{w}$ 的序列对数概率；
- 降低未获选回答 $y^{l}$ 的序列对数概率。

样本权重 $\sigma(-z_\theta)$ 取决于当前偏好间隔。当模型仍偏向 rejected 回答，或尚未建立足够间隔时，该权重较大；当模型已经以较大间隔正确排序时，权重趋近于 0。DPO 因而具有“重点修正当前违反偏好的样本”的自适应加权性质。

上述解释针对参数空间中的局部梯度方向。由于 chosen 与 rejected 共享模型参数，且语言模型概率必须归一化，一次实际更新后每条序列概率如何变化还受到参数耦合、batch 中其他样本和优化器状态影响，不能把两项梯度机械理解为独立修改两个概率值。

### 6.2 参考策略的作用

若只比较 $\log\pi_\theta(y^{w}\mid x)-\log\pi_\theta(y^{l}\mid x)$ ，目标只要求当前策略区分两个回答。DPO 进一步减去参考策略的对应差值，实际比较的是：

> 当前策略相对于参考策略，是否额外提高了 chosen 相对 rejected 的优势。

参考策略通常是 DPO 开始前的 SFT 模型，并在整个训练过程中冻结。当前策略也常由同一 SFT checkpoint 初始化，此时训练初始阶段 $m_\theta\approx0$ ，每个偏好标签的预测概率约为 $0.5$ 。随着训练推进，策略在偏好数据支持的方向上偏离参考模型。

参考策略不是旧策略。它不随每轮训练同步更新，也不负责生成 on-policy rollout；它是定义隐式奖励和行为锚点的固定分布。若错误地持续把参考模型更新为当前模型，DPO 目标的语义将发生改变。

基础 DPO 损失并没有另加一个逐 batch 计算的显式 KL 项，也不保证有限样本训练后的真实策略 KL 低于某个阈值。参考策略约束来自上述奖励重参数化和偏好 logit；实际偏离程度仍需单独监控。

### 6.3 系数 $\beta$ 的解释

在原始 RL 目标中， $\beta$ 是 KL 惩罚系数；在 DPO 损失中，它又是偏好 logit 的缩放系数。二者来自同一个推导。

对于固定的目标偏好概率，较小的 $\beta$ 需要更大的对数概率比间隔才能形成相同 logit，因此通常允许策略相对参考策略移动得更远；较大的 $\beta$ 通常产生更保守的策略。实际训练结果还会受到学习率、训练轮数、数据可分性和模型容量影响，不能只根据 $\beta$ 单独预测最终 KL。

### 6.4 与 SFT 梯度的区别

若只对 chosen 回答做 SFT，目标为 $-\log\pi_\theta(y^{w}\mid x)$ 。它会提高 chosen 的似然，却没有显式利用 rejected 回答，也没有衡量相对参考策略的变化。DPO 同时使用一对回答，并根据当前间隔动态调整梯度权重。

因此，“在 chosen 数据上继续 SFT”并不等价于 DPO。反过来，DPO 也不保证 chosen 回答的绝对概率在每次更新后都增加；它直接约束的是 chosen 与 rejected 的**相对隐式奖励差**。

## 7. 训练算法与实现语义

### 7.1 基础训练算法

基础 DPO 的输入包括偏好数据集 $\mathcal{D}$ 、冻结参考策略 $\pi_{\mathrm{ref}}$ 、初始策略参数 $\theta$ 和系数 $\beta$ 。一个训练迭代包含以下步骤：

1. 从 $\mathcal{D}$ 采样一批 $(x,y^{w},y^{l})$ ；
2. 用当前策略计算 chosen 与 rejected 的序列对数概率；
3. 在无梯度模式下，用参考策略计算相同序列的对数概率；
4. 构造相对参考策略的 chosen / rejected 对数概率比；
5. 计算 $-\log\sigma(\beta m_\theta)$ 的 batch 平均；
6. 只对当前策略参数反向传播并更新。

框架无关的伪代码如下：

```python
for batch in preference_dataloader:
    policy_chosen_logps = policy.sequence_logps(batch["prompt"], batch["chosen"])
    policy_rejected_logps = policy.sequence_logps(batch["prompt"], batch["rejected"])

    with no_grad():
        ref_chosen_logps = reference.sequence_logps(batch["prompt"], batch["chosen"])
        ref_rejected_logps = reference.sequence_logps(batch["prompt"], batch["rejected"])

    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps
    logits = beta * (chosen_log_ratios - rejected_log_ratios)
    loss = -logsigmoid(logits).mean()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

实际实现通常使用数值稳定的 `logsigmoid`，而不是先计算 Sigmoid 再取对数。若参考模型在整个训练期间固定，其序列 log probability 可以预先缓存，以换取存储空间并减少重复前向计算。

### 7.2 response mask 与模板一致性

策略模型和参考模型必须使用完全相同的 tokenizer、chat template、截断规则和回答边界。对于每个序列：

- prompt token 只作为条件，不能计入回答 log probability；
- chosen 与 rejected 各自的有效回答 token 需要参与求和；
- padding 必须被 response mask 排除；
- 结束 token 通常应计入；
- 被截断回答需要明确处理，避免把缺失结尾的序列当作完整回答。

若策略与参考模型使用不同模板，它们计算的就不是同一事件的概率，对数概率比失去意义。若训练与推理模板不同，模型在训练中学到的偏好也可能无法在部署时稳定表现。

### 7.3 序列求和与长度效应

基础 DPO 使用完整回答的 log probability，因此 token 对数概率需要求和。将求和改为平均，会相对放大短回答中每个 token 的作用，并得到不同的统计模型。

使用对数概率比而非原始对数概率，能够抵消参考策略本身对常见表达和部分长度模式的偏好，但不能保证彻底消除长度偏差。回答长度可能同时与标注偏好、截断、结束 token、数据生成方式和隐式奖励累积相关。训练与评估时应记录 chosen / rejected 长度分布、生成长度和长度条件下的胜率。

### 7.4 共享前缀与计算

chosen 与 rejected 可能共享较长前缀。理论上，在回答分叉前，两条序列的相同 token 具有相同条件概率，其贡献会在偏好间隔中抵消。工程实现仍需保证两条序列的模板化结果和 mask 精确对齐。可以通过拼接 chosen / rejected 后进行一次批量前向计算提高吞吐，但这只是计算优化，不改变目标函数。

## 8. DPO 成立所依赖的条件

DPO 推导简洁，但其结论依赖若干建模条件。

### 8.1 标量奖励与 Bradley–Terry 假设

基础推导假设偏好可由单一标量奖励差和 Bradley–Terry 链接函数描述。这意味着偏好大体具有一致、可排序的结构。现实标注可能包含偏好循环、多个互相冲突的价值维度、标注者群体差异和上下文依赖；单一隐式奖励只能对这些现象做近似建模。

### 8.2 同一提示词下的可比性

每个偏好对必须共享相同的条件上下文 $x$ 。若 chosen 和 rejected 实际使用了不同 system prompt、历史消息或工具结果，配分函数不再按上述方式抵消，模型学习到的差异也可能来自上下文而非回答质量。

### 8.3 数据覆盖与参考策略支持

离线偏好数据只约束数据覆盖到的提示词和回答区域。对于训练集之外的回答，DPO 依靠模型参数共享来泛化，而不是获得直接监督。参考策略还需要为数据中的回答赋予可计算的非零概率；神经语言模型的 softmax 在数学上通常满足这一点，但低精度下的下溢、错误 mask 或模板不一致仍可能产生无穷对数比。

### 8.4 固定数据与分布偏移

基础 DPO 通常在固定数据集上离线训练。随着策略变化，更新后的策略在评测或部署时可能生成参考策略或数据收集策略很少产生的新回答，但这些区域没有新的偏好标签。DPO 因而避免了在线 RL 的采样成本，却不能自动修复离线数据与当前策略分布之间的偏移。迭代式 DPO 可以重新采样和标注数据，但那是额外的数据闭环，不属于基础目标本身。

### 8.5 偏好标签的可靠性

基础损失把每条记录视为明确的二元胜负。若标签含有大量噪声、平局或弱偏好，模型可能过度拟合偶然差异。数据构建应记录标注准则、一致性、多标注者分歧和候选回答来源；需要表达偏好强度或不确定性时，应选用相应的加权、平滑或概率标签变体，并明确它们已偏离基础 DPO。

## 9. 与其他后训练方法的关系

| 方法 | 主要监督信号 | 是否显式使用 rejected | 是否需要显式奖励模型 | 是否需要在线 rollout | 核心目标 |
| --- | --- | --- | --- | --- | --- |
| SFT | 理想回答 | 否 | 否 | 否 | 最大化参考回答似然 |
| 奖励模型 | 成对偏好 | 是 | 本身就是奖励模型 | 否 | 预测人类偏好 |
| PPO-RLHF | 奖励模型或规则奖励 | 策略更新阶段不直接使用 | 按奖励来源而定 | 是 | 最大化 KL 正则化奖励 |
| DPO | 成对偏好 | 是 | 否 | 否 | 最大化隐式奖励对应的偏好似然 |

DPO 的优势主要来自训练链路简化：不需要训练 critic，不需要在每次参数更新前生成 rollout，也不需要对不可微奖励做策略梯度估计。它仍然需要偏好数据，而偏好数据本身可能来自人工比较、模型评审、规则验证或其他反馈机制。因此，“不需要奖励模型”不等于“不需要质量判断或数据成本”。

DPO 与 PPO 也不是简单的“监督学习 versus 强化学习”二分。DPO 的目标由 KL 正则化 RL 问题推导而来，但采用离线偏好似然进行优化；PPO [6] 则显式地从当前策略采样并优化奖励。两者在数据覆盖、计算成本和对奖励信号的访问方式上不同。后续理论工作还表明，不同偏好优化算法可以从链接函数、正则化和偏好概率估计等共同要素下统一分析 [7]。

## 10. 数据设计与训练实践

### 10.1 偏好对质量

一个有效偏好对不仅要给出胜负，还应让胜负原因与训练目标一致。至少需要检查：

1. **同条件比较**：两条回答使用完全相同的 prompt 和可见上下文；
2. **标签正确性**：chosen 确实在目标维度上优于 rejected；
3. **差异可学习**：偏好不是由隐藏元数据、候选顺序或模板泄漏决定；
4. **难度适中**：既包含明显错误，也包含高质量回答之间的细粒度比较；
5. **来源多样**：避免 chosen 总来自同一模型或固定风格，使模型学到来源特征；
6. **长度去偏**：检查标注者是否系统性偏爱更长或更短的回答；
7. **数据隔离**：训练偏好对不能泄漏到验证集和测试集。

若 chosen 与 rejected 都很差，或二者都很好，DPO 仍会推动 chosen 与 rejected 的隐式奖励差增大；两者的绝对值则可能同时上升或下降。成对偏好只提供局部排序，不提供“回答是否达到绝对质量门槛”的信息。实践中常先用高质量 SFT 建立基本能力，再用 DPO 调整偏好。

### 10.2 初始化与参考模型

常见配置是让当前策略和参考策略都从同一个 SFT checkpoint 开始，随后冻结参考策略。这样做使训练从可用的指令模型出发，并让隐式奖励表示“相对该 SFT 行为的改进”。

若策略初始化与参考模型相差很大，初始偏好 logit $\beta m_\theta$ 的绝对值可能很大：较大的正 logit 会让对应样本的梯度接近饱和，较大的负 logit 则会产生较强的纠错梯度。若参考模型本身质量很差，KL 锚点也可能限制最终性能。参考策略的选择属于目标定义，而不只是节省显存的工程选项。

### 10.3 过拟合与可分数据

当训练偏好几乎完全可分时，逻辑损失可能持续推动偏好间隔增大，而训练准确率早已饱和。这可能表现为训练 loss 继续下降、策略 KL 持续上升，但生成质量不再改善甚至退化。应使用独立验证集、生成式评估和能力回归决定早停，而不是只追求更低的 DPO loss。

常见控制手段包括减小训练轮数或学习率、调节 $\beta$ 并根据实测 KL 验证效果、改善标签质量、加入标签平滑，以及混合适量 SFT 目标。后两种做法会形成 DPO 变体或复合目标，应在实验报告中明确公式与权重。

## 11. 评估方法

DPO 的训练损失只衡量模型对给定离线偏好对的拟合程度。完整评估至少包括以下层面：

1. **留出偏好拟合**：在验证偏好对上报告 loss、排序准确率和偏好间隔分布；
2. **生成式胜率**：在未见 prompt 上生成回答，与参考模型或基线模型进行盲评；
3. **任务指标**：根据领域报告准确率、代码测试通过率、事实性或格式合规率；
4. **行为漂移**：监控当前策略相对参考策略的序列 KL 估计、生成长度和风格变化；
5. **能力回归**：检查通用问答、指令遵循、多语言、安全性和原有领域能力；
6. **偏差审计**：按回答长度、prompt 类别、候选来源和标注者群体分层统计结果。

偏好分类准确率不是充分指标。只要间隔符号正确，准确率就不再区分置信度；它也不检验模型自由生成时会访问哪些回答。自动裁判可能具有位置、长度和风格偏差，因此关键结论应结合人工盲评或独立可验证指标，并报告置信区间。

建议训练时至少记录：DPO loss、chosen reward、rejected reward、reward margin、偏好准确率、策略与参考策略的 log-ratio、生成长度以及梯度范数。框架日志中的 `reward` 往往指 $\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 这一隐式奖励，不一定是外部奖励模型分数。

## 12. 常见问题与排查方向

| 现象 | 常见原因 | 优先检查 |
| --- | --- | --- |
| loss 长期接近 $\log 2$ | 参数未更新或梯度断开；两条回答相同；log-ratio 恒为 0 | 梯度、优化器、解码后的样本、四个 log probability |
| loss 下降但生成效果不变 | 学到了离线排序但未改变常见生成区域；更新过弱 | 生成式评估、KL、偏好间隔 |
| loss 很快接近 0 且 KL 持续增大 | 数据易分；训练过久； $\beta$ 或正则约束不合适 | 验证 loss、训练轮数、策略 KL |
| 输出明显变长 | 数据存在长度偏好；序列聚合或截断处理不当 | 长度分布、结束 token、length-conditioned win rate |
| 对数概率出现 `nan` 或无穷 | padding / prompt 未屏蔽；低精度下溢；模板不一致 | response mask、log-softmax、tokenizer |
| 更换 batch size 后结果变化很大 | 损失归一化、梯度累积或学习率缩放不一致 | 每个偏好对的标量 loss 与有效样本数 |
| chosen reward 与 rejected reward 同时下降 | DPO 主要优化相对间隔，参数耦合会改变绝对概率 | reward margin、生成质量、策略 KL |
| 验证偏好准确率高但人评下降 | chosen / rejected 方向错误；标注偏差、裁判泄漏或离线过拟合 | 标签方向、独立 prompt、盲评、候选来源分层 |

排查时应先取一个极小 batch，逐条打印 prompt、chosen、rejected、四个序列 log probability 和最终 logit。大多数“代码能运行但目标错误”的问题来自回答边界、模板、标签方向或 mask，而不是优化器本身。

## 13. 理论与方法边界

DPO 显著简化了偏好优化，但并不自动解决对齐中的所有问题：

- **奖励函数被隐式参数化。** DPO 不训练独立奖励模型，但策略—参考策略的 log-ratio 承担了隐式奖励的角色；偏好数据仍然定义模型要追求的价值，数据偏差也会被策略学习；
- **离线数据限制覆盖范围。** 模型可能在未标注区域产生新的失败模式，基础 DPO 不会主动探索并纠正它们；
- **序列级监督的信用分配较粗。** 一个标量偏好无法指出回答中究竟哪些 token 或推理步骤更好；
- **二元标签丢失信息。** 平局、偏好强度、标注者不确定性和多目标权衡不能由基础损失完整表达；
- **似然目标可能过度优化。** 在有限且可分的数据上，偏好间隔可持续增大，不代表真实质量持续提高；
- **参考策略决定正则中心和有效重分配成本。** 参考概率很低的区域需要更大的对数概率比变化；只有当参考概率严格为 0 时，支持集约束才使该区域不可达。DPO 更擅长在已有模型能力附近重新分配概率，而不是凭偏好对创造参考模型从未具备的知识或推理能力；
- **理论等价不等于有限训练完全等价。** 闭式推导针对理想奖励、完整分布和最优策略；有限数据、函数逼近和非凸优化下，DPO 与显式奖励模型加 RL 可能得到不同结果。

因此，DPO 最适合被理解为一种计算简洁、统计假设明确的离线偏好优化方法，而不是对 PPO、奖励建模或在线数据闭环的无条件替代。

## 14. 结论

DPO 的核心可以归纳为以下逻辑链条：

1. 使用标量奖励与 Bradley–Terry 模型描述成对偏好；
2. KL 正则化奖励最大化存在相对参考策略的闭式最优解；
3. 由最优策略反解奖励，并代入偏好概率；
4. 同一提示词下不可计算的配分函数在奖励差中抵消；
5. 最终得到仅依赖当前策略、参考策略和偏好对的二元交叉熵损失。

这一推导解释了为什么语言模型可以“充当自己的奖励模型”： $\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ 在相差一个只依赖提示词的加性常数意义下表示隐式奖励。DPO 以较低工程复杂度实现了偏好优化，但效果仍取决于参考策略、偏好数据覆盖、标签可靠性、序列概率实现和独立生成式评估。理解这些条件，比机械套用一个损失函数更重要。

## 附录 A：从拉格朗日乘子到最优策略

固定提示词 $x$ ，在约束 $\sum_y\pi(y\mid x)=1$ 下最大化 $J_x(\pi)$ 。定义拉格朗日函数

$$
\mathcal{F}(\pi,\lambda)=\sum_y\pi(y\mid x)r(x,y)-\beta\sum_y\pi(y\mid x)\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\lambda(x)\left(\sum_y\pi(y\mid x)-1\right)
$$

对 $\pi(y\mid x)$ 求偏导并令其为 0：

$$
\frac{\partial\mathcal{F}}{\partial\pi(y\mid x)}=r(x,y)-\beta\left(\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+1\right)+\lambda(x)=0
$$

于是

$$
\pi(y\mid x)=\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right)\exp\left(\frac{\lambda(x)}{\beta}-1\right)
$$

最后一个指数因子与 $y$ 无关。利用归一化约束，有

$$
\exp\left(\frac{\lambda(x)}{\beta}-1\right)=\frac{1}{\sum_y\pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)}=\frac{1}{Z(x)}
$$

因此得到

$$
\pi_r^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp\left(\frac{r(x,y)}{\beta}\right)
$$

当 $\pi$ 限制在 $\pi_{\mathrm{ref}}$ 的支持集内、 $\beta\gt0$ 且 $Z(x)$ 有限时， $J_x(\pi)$ 中的负 KL 项关于分布 $\pi$ 严格凹，因此该驻点给出唯一的分布最优解。

## 附录 B：数值检查示例

假设某个偏好对的序列对数概率如下：

| 模型 | chosen | rejected |
| --- | ---: | ---: |
| 当前策略 $\pi_\theta$ | $-8.0$ | $-10.0$ |
| 参考策略 $\pi_{\mathrm{ref}}$ | $-8.5$ | $-9.5$ |

chosen 的对数概率比为 $-8.0-(-8.5)=0.5$ ，rejected 的对数概率比为 $-10.0-(-9.5)=-0.5$ ，所以偏好间隔为 $m_\theta=1.0$ 。若 $\beta=0.1$ ，则偏好 logit 为 $0.1$ ，模型预测 chosen 获胜的概率为 $\sigma(0.1)\approx0.525$ ，单样本损失约为 $0.644$ 。

这个例子说明，DPO 比较的不是当前策略内部 chosen 与 rejected 的原始概率差，而是二者**相对参考策略的改善差**。

## 参考文献

1. Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290). NeurIPS, 2023.
2. Ouyang, L., et al. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS, 2022.
3. Ziegler, D. M., et al. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593). 2019.
4. Christiano, P. F., et al. [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). NeurIPS, 2017.
5. Bradley, R. A., and Terry, M. E. [Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons](https://doi.org/10.2307/2334029). Biometrika, 1952.
6. Schulman, J., et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
7. Azar, M. G., et al. [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036). AISTATS, 2024.
