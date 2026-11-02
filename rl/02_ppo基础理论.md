# PPO（Proximal Policy Optimization）基础理论

PPO（Proximal Policy Optimization，近端策略优化）是一种策略优化算法：让 agent 用当前策略与环境交互，根据奖励判断哪些动作值得强化，再以相对保守的步幅更新策略。它既可以训练游戏或控制任务中的策略，也可以训练语言模型的生成策略。

建议先阅读 [RL 基础理论](./01_rl基础理论.md)，掌握状态、动作、回报、价值函数、优势和策略梯度，再进入本章。这里先解释 PPO 如何把策略梯度变成一套可反复采样和更新的算法，然后用 LLM 逐 token 生成作为贯穿示例；无需先学习预训练或 SFT 的训练方法。

## 1. PPO 在强化学习中的作用

### 1.1 从“提高好动作的概率”到“控制更新幅度”

策略梯度告诉我们：提高正优势动作的概率，降低负优势动作的概率。但一次梯度更新如果改得太多，之前采集的轨迹就可能无法代表新策略的行为，价值估计和优势信号也会变得不可靠。

PPO 的主要做法是先冻结采样时的旧策略，再用新旧策略的动作概率比和裁剪目标控制更新激励。它允许一批轨迹被有限次复用，同时避免持续奖励过大的有利方向变化。这里的“近端”指策略更新尽量留在采样旧策略附近，并不是空间距离或任务步骤的远近。

先记住训练主线，后面的公式都围绕这条主线展开：

```text
旧策略采样轨迹 → 计算奖励和优势 → 用裁剪目标更新策略
       ↑                                      ↓
       └──────── 同步旧策略并重新采样 ──────────┘
```

### 1.2 在语言模型中的应用

很多语言任务很难用唯一的“标准答案”表达。例如，同一个问题可能有多个正确回答，而我们更偏好其中更有帮助或格式更规范的一个。可以让模型先生成回答，再用 PPO 根据奖励调整生成策略。奖励可以来自人工偏好训练出的奖励模型，也可以来自单元测试、数学答案校验器、格式检查器或多个信号的组合。因此，**PPO 是策略优化算法，不等同于 RLHF，也不要求奖励一定来自人类偏好。**

经典 RLHF-PPO 常从已经完成监督微调（SFT）的模型开始。只需把它理解为“已有基本回答能力的初始策略”，它通常有两个作用：

- 初始化待训练的策略，使探索从一个已经能够正常对话的模型开始；
- 作为冻结的参考策略，约束训练后的模型不要偏离初始行为太远。

若直接从不具备基本指令遵循能力的基座模型开始，绝大多数采样回答可能都无法获得有意义的奖励，训练会困难得多。

## 2. 把语言生成写成强化学习问题

### 2.1 状态、动作与轨迹

给定提示词 $x$ ，模型逐 token 生成回答 $y=(y_1,\ldots,y_T)$ 。这个过程可以写成一个有限时长的序列决策问题：

| 强化学习概念 | 在语言模型中的含义 |
| --- | --- |
| 环境初始条件 | 提示词 $x$ |
| 状态 $s_t$ | 提示词和已经生成的回答前缀，即 $s_t=(x,y_{\lt t})$ |
| 动作 $a_t$ | 在位置 $t$ 选择的下一个 token，即 $a_t=y_t$ |
| 策略 $\pi_\theta(a_t\mid s_t)$ | 参数为 $\theta$ 的语言模型给出的 next-token 概率 |
| 轨迹 $\tau$ | 从提示词开始直到结束 token 或长度上限的完整生成过程 |
| 奖励 $r_t$ | 对动作或回答质量的反馈 |
| 回报 $R_t$ | 从位置 $t$ 开始能够获得的累计折扣奖励 |

回答的生成概率仍然遵循自回归分解：

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{\lt t})
$$

SFT 在给定参考回答上提高 token 概率；强化学习则从当前策略采样回答，根据奖励判断这些回答是否值得提高概率。两者都会更新语言模型参数，但数据来源和优化信号不同。

### 2.2 序列级奖励与 token 级决策

很多 LLM 任务只在完整回答生成后给出一个序列级分数。例如，奖励模型读入 $(x,y)$ 后输出标量 $r_\phi(x,y)$ 。然而，策略在每个 token 位置都做了一次决策，因此必须把最终结果转化为各位置的学习信号。这就是**信用分配（credit assignment）**问题：回答获得高分，究竟是哪些 token 决策起了作用？

最简单的做法是把任务奖励只放在最后一个有效 token 上，前面位置的外部任务奖励均为 0；再通过回报和优势估计，把终局信息传回前面的状态。若有过程奖励，也可以在中间步骤提供反馈，但奖励定义必须与 token 或推理步骤边界正确对齐。

## 3. PPO 训练中的几个模型

一套典型的 RLHF-PPO 系统包含以下角色：

| 角色 | 常用符号 | 是否更新 | 作用 |
| --- | --- | --- | --- |
| 策略模型（actor） | $\pi_\theta$ | 是 | 生成回答，也是最终要得到的模型 |
| 旧策略 | $\pi_{\mathrm{old}}$ | 一轮更新期间冻结 | 生成当前批次数据，并作为 PPO 概率比的分母 |
| 参考模型 | $\pi_{\mathrm{ref}}$ | 否 | 提供 KL 约束的行为锚点，通常由初始 SFT 模型冻结得到 |
| 奖励模型 | $r_\phi$ | PPO 阶段通常冻结 | 为完整回答或中间步骤打分 |
| 价值模型（critic） | $V_\psi$ | 是 | 预测未来回报，帮助估计优势并降低策略梯度方差 |

这里最容易混淆的是旧策略与参考模型：

- $\pi_{\mathrm{old}}$ 是当前策略在采样时的快照。每轮采样后暂时固定，下一轮通常会同步为新的策略；它服务于 PPO 的重要性采样概率比。
- $\pi_{\mathrm{ref}}$ 是训练开始时选定的长期锚点，通常在整个 PPO 阶段保持冻结；它服务于 KL 正则。

二者有时初始权重相同，但训练语义完全不同，不能因为都“冻结”就混为一个模型。

## 4. PPO 真正想优化什么

### 4.1 奖励目标

忽略 KL 约束时，策略优化的抽象目标是最大化采样回答的期望任务奖励：

$$
J(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\left[r_\phi(x,y)\right]
$$

其中， $\mathcal{D}$ 是提示词分布， $y\sim\pi_\theta(\cdot\mid x)$ 表示回答由当前策略采样， $r_\phi(x,y)$ 是奖励模型或规则给出的分数。与 SFT 不同，这里没有要求模型复现某个固定参考答案；只要回答能获得高奖励，就可能被策略强化。

### 4.2 为什么要加入 KL 约束

若只追求奖励最大化，策略可能找到奖励函数的漏洞，生成在分数上很高、实际质量却很差的文本。这称为 reward hacking。策略也可能迅速偏离原有语言分布，导致表达退化、重复或通用能力下降。

常见做法是在任务奖励中减去相对参考策略的 KL 惩罚。序列级正则化目标可以写成：

$$
J_{\mathrm{KL}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\left[r_\phi(x,y)-\beta\sum_{t=1}^{T}\left(\log\pi_\theta(y_t\mid s_t)-\log\pi_{\mathrm{ref}}(y_t\mid s_t)\right)\right]
$$

其中， $\beta$ 是 KL 系数。括号中的 sampled log-ratio 是在实际采样 token 上对 KL 的估计项；单个 token 的值可以为负，但对当前策略的动作分布取期望后得到非负的 KL 散度。

直观上：

- $\beta$ 太小，策略更容易偏离参考模型并利用奖励漏洞；
- $\beta$ 太大，策略几乎不敢改变，任务奖励难以上升；
- 实践中可以使用固定 $\beta$ ，也可以根据目标 KL 自适应调整。

KL 惩罚通常被拆成 token 级奖励：

$$
r_t^{\mathrm{KL}}=-\beta\left(\log\pi_{\mathrm{old}}(a_t\mid s_t)-\log\pi_{\mathrm{ref}}(a_t\mid s_t)\right)
$$

然后在回答最后一个有效位置再加上任务奖励。这里使用 $\pi_{\mathrm{old}}$ 是因为当前 rollout 由旧策略生成；不同实现也可能直接在优化目标中加入 KL 项，具体公式和日志含义需要结合框架确认。

> **PPO clipping 与参考模型 KL 不是同一个约束。** 前者削弱当前策略相对采样旧策略改得过多的收益，后者惩罚整个训练过程相对参考模型的漂移。二者都是优化中的控制手段，不能理解成必然满足的硬距离界；只启用其中一个，也不能自动得到另一个的效果。

## 5. 价值模型、回报与优势

### 5.1 回报

设经过任务奖励和 KL 惩罚整形后的 token 奖励为 $r_t$ ，折扣因子为 $\gamma$ 。对于在位置 $T$ 真正结束的完整轨迹，从位置 $t$ 开始的累计回报为：

$$
R_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
$$

在有限长度的语言生成任务中，常取 $\gamma=1$ ，因为回答前面的 token 也应对最终结果负责。不过，公式保留 $\gamma$ 能与一般强化学习记号保持一致。

价值模型 $V_\psi(s_t)$ 预测从状态 $s_t$ 出发的期望回报。它不是评价某个 token 在语言上是否正确，而是在当前策略和奖励定义下预测“从此前缀继续生成，预计能得到多少回报”。

### 5.2 优势函数

仅知道某条轨迹回报高还不够，因为不同提示词本身的奖励难度不同。优势函数比较某个动作的实际结果与当前状态下的平均预期：

$$
A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)
$$

优势为正，说明该动作比状态基线预期更好，应提高其概率；优势为负，说明它比预期更差，应降低其概率。减去只依赖状态的基线不会改变策略梯度的期望，却能显著降低方差。

### 5.3 GAE 优势估计

PPO 常用广义优势估计（Generalized Advantage Estimation，GAE）。先定义一步时序差分残差，其中 $\chi_t=1$ 表示执行动作 $a_t$ 后任务真正终止，否则 $\chi_t=0$ ：

$$
\delta_t=r_t+\gamma(1-\chi_t)V_\psi(s_{t+1})-V_\psi(s_t)
$$

再将未来残差加权累积：

$$
\hat A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\delta_{t+l}
$$

其中， $\lambda\in[0,1]$ 控制偏差与方差的权衡。 $\lambda$ 较小，更依赖价值模型的一步预测，方差较低但价值误差带来的偏差可能更大； $\lambda$ 较大，更接近完整轨迹回报，完整轨迹上的 bootstrap 偏差较小，但方差通常更高。上式只在同一条有效轨迹内累加，不能越过回答边界或进入 padding。

这里必须区分两种“结束”：

- **任务真正终止**：例如生成结束 token，或者任务把长度上限明确规定为 episode 的结束。此时终止后的未来回报为 0，最后一步取 $\chi_T=1$ ，不从下一个状态 bootstrap。
- **采集片段被截断**：例如为了分批采样，只收集了某个仍可继续的任务的前 $T$ 步。此时取 $\chi_T=0$ ，最后一个 TD 残差保留下一个状态的价值，用它估计未采集的未来回报；GAE 的累加仍停在片段边界。

bootstrap 就是“用价值预测补上尚未采集到的未来”。对于第二种情况，截断片段的回报估计可以写成：

$$
\hat R_t^{\mathrm{bootstrap}}=\sum_{k=t}^{T}\gamma^{k-t}r_k+\gamma^{T-t+1}V_\psi(s_{T+1})
$$

不能仅凭“触及 `max_new_tokens`”就决定是否补未来价值：如果任务把超长回答视为结束并判罚，它属于第一种情况；如果只因采集预算停止一个本来仍会继续的 episode，它属于第二种情况。错误地把后者的未来价值清零，会让模型把“暂时没有采集到奖励”误学成“未来没有奖励”。

价值模型通常用回报目标 $\hat R_t$ 做回归：

$$
\mathcal{L}_{V}(\psi)=\mathbb{E}_t\left[\left(V_\psi(s_t)-\hat R_t\right)^2\right]
$$

采用 GAE 时，常用回报目标为：

$$
\hat R_t=\hat A_t+V_{\mathrm{old}}(s_t)
$$

其中， $V_{\mathrm{old}}$ 表示采样时价值模型给出的冻结预测。这一目标通常称为 lambda-return，并不总等于完整的 Monte Carlo 回报。优势和回报目标在本轮 minibatch 更新中保持固定。常见实现会进一步裁剪价值更新，以减少 critic 在一个批次内变化过大。

## 6. PPO 的裁剪策略目标

### 6.1 从策略梯度到新旧策略概率比

最直接的策略梯度根据采样动作的 log probability 和优势更新策略。但 PPO 希望对一批 rollout 做有限次 minibatch 更新：采样只发生一次，当前策略却持续变化，旧样本中的动作频率就不再等于当前策略的动作概率。

为处理这个差异，PPO 使用动作概率比衡量当前策略相对旧策略的变化：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right)
$$

刚开始更新时，当前策略与旧策略相同，因此 $\rho_t(\theta)=1$ 。随着同一批数据被训练多个 minibatch epoch，二者逐渐产生差异。例如旧策略给已采样动作的概率是 0.2，新策略给出 0.3，那么概率比是 1.5，表示该动作在新策略下比采样时更常出现。

概率比来自**重要性采样**的思想：在固定状态 $s_t$ 上，把旧策略抽到的动作按新旧概率比加权，可以校正动作分布变化。要求旧策略在新策略可能选择的动作上也有非零概率。未裁剪的代理目标是：

$$
\mathcal{L}_{\mathrm{surrogate}}(\theta)=\mathbb{E}_{(s_t,a_t)\sim\pi_{\mathrm{old}}}\left[\rho_t(\theta)\hat A_t\right]
$$

在新旧策略相同的起点，对这个目标求梯度，会得到以旧策略样本和优势估计表达的策略梯度方向。这里仍使用旧策略采集的状态和旧策略下估计的优势；单步动作概率比没有把整个轨迹的状态分布也变成当前策略的分布，所以它是旧策略附近的**代理目标**，不是任意大幅更新后真实回报的精确表达式。

旧 log probability 必须保留采样时的值，优势与回报目标也应固定。如果每次更新都把分母改成当前模型，概率比会被错误地重置为 1，PPO 就失去了衡量策略变化的参照。

### 6.2 裁剪代理目标

PPO 的核心是最大化下面的裁剪代理目标：

$$
\mathcal{L}_{\mathrm{clip}}(\theta)=\mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat A_t,\,\mathrm{clip}\left(\rho_t(\theta),1-\varepsilon,1+\varepsilon\right)\hat A_t\right)\right]
$$

其中， $\varepsilon$ 是裁剪范围。这个目标可以分两种情况理解：

- 当 $\hat A_t\gt0$ 时，希望增大动作概率，但比例超过 $1+\varepsilon$ 后，不再从代理目标获得额外收益；
- 当 $\hat A_t\lt0$ 时，希望减小动作概率，但比例低于 $1-\varepsilon$ 后，不再因继续减小而获得额外收益。

取两个分支的较小值，是为了构造一个保守目标，削弱“朝有利方向一次改得过头”的激励。PPO 因此比直接使用未裁剪策略梯度更稳定，也比带硬约束的 TRPO 更容易实现。

需要注意，clipping 裁剪的是代理目标的一个分支，**没有把模型概率或实际概率比强制限制在裁剪区间内**，也不保证更新后策略的真实 KL 一定低于某个阈值。未采样动作、共享参数引起的联动，以及多次梯度更新都可能让实际策略变化超出直觉范围。因此仍应监控新旧策略 KL、clip fraction 和概率比异常值，必要时提前停止当前批次的更新。

### 6.3 手算一次 clipping

设 $\varepsilon=0.2$ ，因此裁剪区间是 $[0.8,1.2]$ 。每一行代表一个采样动作在某次更新后的情况：

| 优势 | 概率比 | 未裁剪项：概率比 × 优势 | 裁剪项：裁剪后概率比 × 优势 | 取较小值后的目标 |
| --- | --- | --- | --- | --- |
| 2 | 1.1 | 2.2 | 2.2 | 2.2 |
| 2 | 1.5 | 3.0 | 2.4 | 2.4 |
| -2 | 0.9 | -1.8 | -1.8 | -1.8 |
| -2 | 0.5 | -1.0 | -1.6 | -1.6 |
| -2 | 1.5 | -3.0 | -2.4 | -3.0 |

正优势动作的概率比从 1.2 增到 1.5，或者负优势动作的概率比从 0.8 降到 0.5，都不会继续改善该样本的目标值。最后一行则是“负优势动作反而更常被选择”，目标仍保留 -3.0，继续惩罚这次错误方向的变化。因此 clipping 不是把所有超界变化一律屏蔽。

这些数值是需要**最大化**的代理目标；代码里的 policy loss 通常取其相反数并最小化。若把表格中的负值误当成“越负越好的 loss”，会把学习方向颠倒。

### 6.4 总训练损失

将策略目标、价值损失和可选的熵奖励组合，可以写成：

$$
\mathcal{L}_{\mathrm{total}}(\theta,\psi)=-\mathcal{L}_{\mathrm{clip}}(\theta)+c_V\mathcal{L}_{V}(\psi)-c_H\mathcal{H}(\pi_\theta)
$$

其中， $c_V$ 和 $c_H$ 是权重， $\mathcal{H}(\pi_\theta)$ 是策略熵。最前面的负号是因为优化器通常最小化 loss，而策略目标需要最大化。LLM RLHF 实现不一定显式加入熵奖励，因为采样温度和参考模型 KL 已能影响探索；不能仅凭公式判断某个框架一定包含该项。

## 7. 一轮 PPO 训练发生了什么

```text
从提示词数据集中采样一批 prompt
  ↓
旧策略生成回答，并保存每个有效 token 的 log probability
  ↓
奖励模型或规则计算任务奖励
参考模型计算 log probability，形成 KL 惩罚
价值模型预测各位置价值
  ↓
计算 token 奖励、回报和 GAE 优势
  ↓
打乱 rollout，进行若干次 minibatch 更新
  ├─ 更新策略模型：PPO clipped objective
  └─ 更新价值模型：value loss
  ↓
丢弃旧 rollout，同步旧策略，重新采样
```

框架无关的伪代码如下：

```python
for prompts in prompt_dataloader:
    with no_grad():
        responses, response_mask, terminated = old_policy.generate(prompts)
        old_log_probs = old_policy.log_probs(prompts, responses)
        ref_log_probs = reference_policy.log_probs(prompts, responses)
        task_rewards = reward_fn(prompts, responses)
        old_values = value_model.values(prompts, responses)
        bootstrap_values = value_model.bootstrap_values(
            prompts, responses, terminated
        )

        token_rewards = build_rewards(
            task_rewards=task_rewards,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
        )
        advantages, returns = compute_gae(
            token_rewards,
            old_values,
            bootstrap_values=bootstrap_values,
            terminated=terminated,
            response_mask=response_mask,
        )

    rollout = pack_rollout(
        prompts, responses, old_log_probs, old_values,
        advantages, returns, response_mask
    )
    for minibatch in repeat_and_shuffle(rollout):
        new_log_probs = policy.log_probs(minibatch)
        new_values = value_model.values(minibatch)

        ratio = (new_log_probs - minibatch.old_log_probs).exp()
        policy_loss = clipped_policy_loss(
            ratio, minibatch.advantages, response_mask=minibatch.response_mask
        )
        value_loss = clipped_value_loss(
            new_values, minibatch.returns, minibatch.old_values,
            response_mask=minibatch.response_mask
        )

        optimize(policy_loss + value_coefficient * value_loss)

    old_policy.load_state_dict(policy.state_dict())
```

这里的函数是框架无关的流程占位，并非可直接运行的 API。`bootstrap_values` 对真正终止的回答应为 0，对仅被采集预算截断的片段应取最后动作之后状态的价值。伪代码展示了 response mask 和终止信息的传递，省略了其内部实现、attention mask、分布式通信、梯度累积和数值稳定处理。真实实现中，所有逐 token 张量必须只在有效回答位置参与计算。

## 8. 为什么 PPO 有效，又为什么训练成本高

PPO 适合 LLM 后训练，主要因为：

1. **直接优化生成结果**：奖励可以评价完整回答，而不要求每个输入只有一个参考答案。
2. **更新相对保守**：PPO clipping 抑制过大的有利方向更新；在 LLM RLHF 中，还常用参考模型 KL 控制长期漂移。
3. **优势估计较稳定**：critic 为不同状态提供基线，降低长序列策略梯度的方差。
4. **能够复用一批 rollout**：同一批 on-policy 数据可以做有限次 minibatch 更新，提高采样利用率。

它的主要成本也来自这些组件：

- 策略、参考、奖励和价值四类模型可能同时占用显存；
- rollout 是自回归生成，通常比 SFT 的 teacher forcing 前向计算慢；
- 训练数据必须保持足够 on-policy，旧样本不能无限复用；
- 奖励尺度、KL 系数、价值模型和策略学习率相互影响，调参比 SFT 更敏感。

工程上可以共享部分模型权重、使用 LoRA、量化冻结模型、卸载模型或采用分布式并行，但这些优化不能改变各角色的数学语义。

## 9. 实践中的关键细节

### 9.1 只在回答 token 上优化

prompt token 是策略的条件，不是这次 rollout 中采取的动作。策略损失、价值损失、KL、优势归一化和日志统计通常都应使用 response mask 排除 prompt 与 padding。结束 token 一般属于回答，应保留；超过长度上限但未正常结束的回答，可以单独施加惩罚或记录指标。

### 9.2 奖励与优势的尺度

不同奖励源的数值范围可能差异很大。直接相加会让尺度较大的信号支配训练，因此常需做加权、裁剪、中心化或标准化。优势也常在有效 token 上进行 whitening。分布式训练时必须明确统计量是在单卡、全局 batch 还是每个 prompt 内计算，否则有效超参数会随并行规模改变。

### 9.3 保持 on-policy

PPO 允许对同一批 rollout 做有限次更新，但不意味着可以长期重复训练旧回答。更新 epoch 太多或学习率太大时， $\pi_\theta$ 与 $\pi_{\mathrm{old}}$ 的差距扩大，大量 token 被裁剪，梯度信号会失真，重要性采样方差也会升高。应同时观察 approximate KL、clip fraction 和有效样本比例。

### 9.4 生成配置也是训练配置

温度、top-p、最大回答长度和停止条件决定策略探索到什么数据。采样过于确定，回答缺乏多样性；采样过于随机，奖励噪声和无效回答增多。训练时与评测时可以使用不同的解码配置，但必须分别记录，不能把解码变化误认为模型能力变化。

概率比还要求分母描述实际采样策略。温度缩放和 top-p 过滤会改变采样分布，不能在数学上自动把过滤前的原始 softmax 概率当成过滤后策略的精确概率。落地时应核对框架如何定义 sampling policy 和计算 log probability，尤其避免新旧策略使用不一致的温度或过滤规则。

## 10. 常见问题与排查方向

| 现象 | 常见原因 | 优先检查 |
| --- | --- | --- |
| 奖励快速上升但人工观感下降 | reward hacking；奖励模型分布外失真 | 高奖励样本、各奖励分量、人评 |
| KL 持续增大 | KL 系数过小；学习率或更新 epoch 过大 | KL、概率比、clip fraction |
| KL 很小且奖励不升 | KL 系数过大；优势接近 0；策略更新过弱 | 优势分布、梯度范数、有效 mask |
| value loss 很大或发散 | 奖励尺度剧烈变化；critic 学习率过大；终止状态处理错误 | returns、values、response mask |
| 大量 token 被裁剪 | 当前策略离旧策略过远 | 更新 epoch、学习率、采样后权重同步 |
| 回答越来越长 | 奖励与长度正相关；结束 token 信号不足 | 长度—奖励相关性、截断比例 |
| 回答模式单一 | 探索不足；奖励偏好固定模板；KL 过强 | 生成温度、熵、样本多样性 |
| 显存远超预期 | 多个模型都保留梯度或全精度副本 | 参数冻结、共享策略、量化与卸载配置 |

不要只看总 reward。至少应分别记录任务奖励、KL 惩罚、总奖励、回答长度、正常结束比例、策略 KL、clip fraction、熵、value loss、returns 和 advantages 的分布。

## 11. 如何评估 PPO 模型

评估需要区分“奖励函数认为更好”和“模型真的更好”：

1. **训练稳定性**：观察 reward、KL、value loss、clip fraction、梯度范数和生成长度是否稳定。
2. **目标能力**：在隔离测试集上使用任务指标、规则校验或成对胜率。
3. **奖励泛化**：使用未参与训练的奖励模型、独立规则或人工评价，避免只对训练奖励过拟合。
4. **能力回归**：检查通用问答、指令遵循、安全性、语言质量和原有领域能力。
5. **生成审计**：人工阅读高奖励、低奖励和长度异常的样本，寻找 reward hacking 模式。

比较不同 checkpoint 时，应固定 prompt 集合和解码参数，并报告置信区间或足够大的样本量。奖励均值的小幅变化可能只是采样噪声。

## 12. PPO 的能力边界与 GRPO 的引出

PPO 的 critic 能利用每个状态的信息建立基线，但训练一个与策略规模接近的价值模型代价很高，而且价值估计本身可能不稳定。对于数学、代码等可以为同一个 prompt 批量采样多个回答并自动判分的任务，可以不用单独的 critic，而把同组回答的平均奖励作为基线。这正是 GRPO 的核心思路。

GRPO 保留了“旧策略采样、概率比、clipping、参考模型 KL”等 PPO 家族特征，却用组内相对奖励替代学习得到的价值函数。它减少了显存和训练复杂度，但会引入组内采样成本以及新的方差和偏差问题，详见 [GRPO 基础理论](./03_grpo基础理论.md)。

## 13. 本章小结

- LLM 生成可以看作逐 token 决策过程，完整回答是轨迹，奖励评价生成结果。
- PPO 使用价值模型和 GAE 估计优势，再用新旧策略概率比与 clipped objective 更新策略。
- 旧策略用于约束一次 rollout 上的更新，参考策略用于限制相对初始模型的长期漂移，二者不能混淆。
- KL 约束、PPO clipping、奖励尺度、response mask 和 on-policy 数据共同影响训练稳定性。
- PPO 能直接优化不可微的序列级目标，但需要维护多个模型、在线生成数据，并谨慎防范 reward hacking。

## 14. 学完后自检

进入 [GRPO 基础理论](./03_grpo基础理论.md) 前，先尝试不看公式回答这些问题：

1. 旧策略和参考策略分别在什么时候同步？它们为什么不能混用？
2. 为什么 PPO 要保留采样时的 log probability，而不能每次更新后重新计算分母？
3. 对正优势动作，概率比从 1.2 增到 1.5 时，裁剪目标还能增加吗？实际概率比能否达到 1.5？
4. 为什么完整回答只在最后获得奖励，前面的 token 仍可以获得学习信号？
5. 一个任务尚未结束，只因为采样片段用完而停止，最后一步是否应该把未来价值清零？
6. PPO 的价值模型提供了什么基线？如果不训练它，要从哪里得到比较动作好坏的参照？

第 3 题的答案是“不能增加，但实际概率比可以达到 1.5”；第 5 题的答案是“不应清零，应 bootstrap”。如果能解释其余问题，就已经掌握学习 GRPO 所需的 PPO 主线。

## 参考文献

1. Schulman, J., et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
2. Schulman, J., et al. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR, 2016.
3. Ziegler, D. M., et al. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593). 2019.
4. Stiennon, N., et al. [Learning to Summarize with Human Feedback](https://arxiv.org/abs/2009.01325). NeurIPS, 2020.
5. Ouyang, L., et al. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS, 2022.
6. Bai, Y., et al. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). 2022.
