# PPO（Proximal Policy Optimization）基础理论

PPO（Proximal Policy Optimization，近端策略优化）是一种策略优化算法：让 agent 用当前策略与环境交互，根据奖励判断哪些动作值得强化，再以相对保守的步幅更新策略。它既可以训练游戏或控制任务中的策略，也可以训练语言模型的生成策略。

建议先阅读 [RL 基础理论](./01_rl基础理论.md)，掌握状态、动作、回报、价值函数、优势和策略梯度，再进入本章。这里先解释 PPO 如何把策略梯度变成一套可反复采样和更新的算法，然后用 LLM 逐 token 生成作为贯穿示例；无需先学习预训练或 SFT 的训练方法。

## 1. PPO 在强化学习中的作用

### 1.1 从“提高好动作的概率”到“控制更新幅度”

策略梯度告诉我们：提高正优势动作的概率，降低负优势动作的概率。但一次梯度更新如果改得太多，之前采集的轨迹就可能无法代表新策略的行为，价值估计和优势信号也会变得不可靠。

PPO 的主要做法是先冻结采样时的旧策略，再用新旧策略的动作概率比和裁剪目标控制更新激励。它允许一批轨迹被有限次复用，同时避免持续奖励过大的有利方向变化。这里的“近端”指策略更新尽量留在采样旧策略附近，并不是空间距离或任务步骤的远近。

本章采用常见的 **PPO-Clip** 版本。PPO 原论文还提出自适应 KL 惩罚版本；KL 是衡量两个概率分布差异的散度，该版本用旧策略到当前策略的 KL 控制本轮更新。这与 LLM 配方里相对长期参考模型的 KL 作用不同，参考模型 KL 见第 4.2 节。学习 PPO-Clip 时，先掌握裁剪目标即可，不需要提前展开 TRPO 的优化过程。

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

仅知道某条轨迹回报高还不够，因为不同提示词本身的奖励难度不同。真实优势函数比较的是“指定这个动作后的期望回报”与“当前策略在这个状态的平均期望回报”：

$$
A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)
$$

这里的 $Q^\pi$ 和 $V^\pi$ 都是期望，不能用某个动作一次采样恰好成功或失败来定义真实优势。实际训练只能构造优势估计 $\hat A_t$ ，它会受到采样随机性和价值模型误差的影响。

真实优势为正，说明该动作比状态基线预期更好；为负，说明它比预期更差。策略梯度据此产生提高或降低动作概率的学习信号。减去只依赖状态的合适基线可以降低方差；动作无关基线不改变策略梯度期望，但不保证任意基线都达到最低方差。

### 5.3 先看 critic 的 Monte Carlo 与 TD 目标

把采样时价值模型的预测记为 $V_{\mathrm{old}}(s_t)$ 。下文构造目标时使用这份固定预测，之后更新的当前价值模型仍记为 $V_\psi(s_t)$ 。这两个记号表示同一个 critic 在不同时间的角色，不要求额外训练一个价值网络。

如果已经采集到完整 episode，可以用实际回报做 Monte Carlo（MC）目标：

$$
\hat R_t^{\mathrm{MC}}=R_t
$$

如果只观察了一步转移，可以用即时奖励和下一状态的价值预测做 TD 目标。令 $\chi_t=1$ 表示执行动作 $a_t$ 后任务真正终止，否则 $\chi_t=0$ ：

$$
\hat R_t^{\mathrm{TD}}=r_t+\gamma(1-\chi_t)V_{\mathrm{old}}(s_{t+1})
$$

无论选择哪种回报目标，critic 都可以用平方误差回归。下式中的 $\mathbb{E}_t$ 表示对有效采样位置求平均：

$$
\mathcal{L}_{V}(\psi)=\mathbb{E}_t\left[\left(V_\psi(s_t)-\mathrm{stopgrad}(\hat R_t)\right)^2\right]
$$

`stopgrad` 表示将目标视为常量，梯度仍通过当前预测 $V_\psi(s_t)$ 传播。MC 需要等完整回报，目标随机性可能较大；TD 可以更早更新，但依赖下一状态的价值预测。用已有价值预测补上未采集的未来回报称为 bootstrap，预测不准会影响目标。**MC 不等于离线 RL，TD 也不等于 on-policy**：用什么回报估计、何时更新、数据来自什么策略，是不同的问题。

一些实现还会相对旧预测裁剪价值更新，这是可选的稳定化处理；旧预测和回报目标仍是固定参照，当前价值模型仍需要求导。

### 5.4 一步、两步、三步到 k 步优势

一步 TD 残差是“这一步观察到的回报目标，比原来的状态价值预测好多少”：

$$
\delta_t=r_t+\gamma(1-\chi_t)V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t)
$$

先看同一条采样片段内部的推导，不越过 episode 边界。以下展开中，若最后状态是真正终止状态，它的 bootstrap 价值按 0 处理。一步优势估计就是这个残差：

$$
\hat A_t^{(1)}=\delta_t=r_t+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t)
$$

再多看一步，把两个残差按折扣累加：

$$
\begin{aligned} \hat A_t^{(2)}&=\delta_t+\gamma\delta_{t+1} \\ &=r_t+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t)+\gamma\left(r_{t+1}+\gamma V_{\mathrm{old}}(s_{t+2})-V_{\mathrm{old}}(s_{t+1})\right) \\ &=r_t+\gamma r_{t+1}+\gamma^2V_{\mathrm{old}}(s_{t+2})-V_{\mathrm{old}}(s_t) \end{aligned}
$$

中间状态的价值一正一负抵消，这称为裂项求和（telescoping sum）。同样，三步优势估计为：

$$
\hat A_t^{(3)}=\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2}=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\gamma^3V_{\mathrm{old}}(s_{t+3})-V_{\mathrm{old}}(s_t)
$$

推广到 k 步，得到：

$$
\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}=\sum_{l=0}^{k-1}\gamma^lr_{t+l}+\gamma^kV_{\mathrm{old}}(s_{t+k})-V_{\mathrm{old}}(s_t)
$$

因此，每种步数都对应“观察 k 步实际奖励 + 用尾部价值补未来 - 减去起点基线”。看得越远，使用的实际反馈越多，但也带入更多随机性；只看一步则更依赖 critic 的预测。

### 5.5 有限轨迹上的 lambda-return 与 GAE

从位置 $t$ 到当前采样片段结尾，还剩 $K=T-t+1$ 个有效动作。对应的 k 步回报目标为：

$$
\hat R_t^{(k)}=\hat A_t^{(k)}+V_{\mathrm{old}}(s_t)
$$

GAE（Generalized Advantage Estimation，广义优势估计）不只选择某一个步数，而是用 $\lambda\in[0,1]$ 混合不同跨度。**有限片段不能把无限序列的权重直接截掉**，最后一个完整片段目标要接住剩余权重：

$$
\hat R_t^\lambda=(1-\lambda)\sum_{k=1}^{K-1}\lambda^{k-1}\hat R_t^{(k)}+\lambda^{K-1}\hat R_t^{(K)}
$$

前 $K-1$ 项的权重是“停止在这个步数”的权重，最后一项的权重是“至少看完当前片段”的剩余权重。所有权重之和为 1；当 $K=1$ 时，前面的和为空，唯一目标的权重为 1。将每项都减去同一个起点基线，可得：

$$
\hat A_t^{\mathrm{GAE}}=(1-\lambda)\sum_{k=1}^{K-1}\lambda^{k-1}\hat A_t^{(k)}+\lambda^{K-1}\hat A_t^{(K)}=\sum_{l=0}^{K-1}(\gamma\lambda)^l\delta_{t+l}
$$

这就是常见的 GAE 残差累加公式。它与 lambda-return 的关系为：

$$
\hat R_t^\lambda=\hat A_t^{\mathrm{GAE}}+V_{\mathrm{old}}(s_t)
$$

实践中常简写为 $\hat A_t$ 和 $\hat R_t$ ，分别用于 actor 更新和 critic 回归。两者均在本轮 rollout 更新前计算并固定，不能在每个 minibatch 更新后用新的 critic 重算目标。

### 5.6 逆序递推、真正终止与截断

第 5.5 节的 GAE 残差累加不必为每个位置重新求和，可以从最后一步逆序递推。令 $c_t=1$ 表示下一条转移仍属于同一条有效采样片段，否则为 0：

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}
$$

最后一个有效动作取 $c_T=0$ 。它只意味着“没有更多已采集的残差继续累加”，**不意味着下一状态价值必定为 0**。两种结束语义分别为：

| 情况 | 最后一步终止标记 | 最后一步 bootstrap | GAE 是否跨片段继续累加 |
| --- | --- | --- | --- |
| 任务真正终止 | $\chi_T=1$ | 0 | 否 |
| 任务仍可继续，仅采集片段结束 | $\chi_T=0$ | $V_{\mathrm{old}}(s_{T+1})$ | 否 |

bootstrap 就是“用价值预测补上尚未采集到的未来”。不能仅凭“触及 `max_new_tokens`”就决定是否补未来价值：如果任务把超长回答视为 episode 结束并判罚，它属于真正终止；如果只是采样预算耗尽，任务本来仍会继续，则要保留尾部价值。

批量计算还必须排除 prompt 与 padding，不能将下一条回答的价值或残差接到当前回答后面。截断时使用的是**最后一个实际动作之后的状态价值**，而不是最后一个 padding 位置的价值；采样时保存的 critic 预测和尾部价值都应冻结。

### 5.7 lambda 的两个端点与偏差、方差

当 $\lambda=0$ 时，仅使用一步 TD 残差：

$$
\hat A_t^{\lambda=0}=\delta_t
$$

当 $\lambda=1$ 时，全部残差裂项相消，得到：

$$
\hat A_t^{\lambda=1}=\sum_{l=0}^{K-1}\gamma^lr_{t+l}+\gamma^KV_{\mathrm{old}}(s_{T+1})-V_{\mathrm{old}}(s_t)
$$

完整 episode 真正终止时，尾部价值为 0，因此它等于实际回报 $R_t$ 减去状态基线。即使 critic 的起点预测不准，这个动作无关基线也不改变策略梯度的期望；但该估计量本身不一定是“真实优势数值”的条件无偏估计。仅采集了截断片段时，尾部 bootstrap 误差仍会影响梯度，不能沿用完整回报的结论。

较小的 $\lambda$ 通常使用更多短步价值预测，较大的 $\lambda$ 通常使用更多长程实际反馈，这是调节估计方式的方向。**不能保证对任意环境、奖励与 critic，偏差或方差都随 $\lambda$ 单调变化**；误差、奖励及残差之间的相关性也会影响结果。折扣因子 $\gamma$ 还定义时间偏好的目标，不能把它只当成降方差参数，更不能声称改变折扣一定不影响动作排序。正式的梯度无偏条件见附录 A。

### 5.8 一条三步轨迹的完整手算

为了看清折扣和 bootstrap，取一个一般 RL 例子：即时奖励为 `[0, 0, 1]`， $\gamma=0.9$ ，采样时四个状态的价值预测为 `[0.2, 0.4, 0.6, 0]`，最后一步真正终止。三个 TD 残差为：

$$
\delta_1=0+0.9\times0.4-0.2=0.16,\qquad\delta_2=0+0.9\times0.6-0.4=0.14,\qquad\delta_3=1-0.6=0.4
$$

起点的一步、两步和三步优势分别为：

$$
\hat A_1^{(1)}=0.16,\qquad\hat A_1^{(2)}=0.16+0.9\times0.14=0.286,\qquad\hat A_1^{(3)}=0.16+0.9\times0.14+0.9^2\times0.4=0.61
$$

取 $\lambda=0.8$ ，三个步数的权重分别为 `0.2、0.16、0.64`。起点的 GAE 为：

$$
\hat A_1=0.2\times0.16+0.16\times0.286+0.64\times0.61=0.46816
$$

也可以逆序算出同样的结果：

$$
\hat A_3=0.4,\qquad\hat A_2=0.14+0.72\times0.4=0.428,\qquad\hat A_1=0.16+0.72\times0.428=0.46816
$$

对应的 lambda-return 是 `[0.66816, 0.828, 1]`。它们由优势加回固定状态价值得到，区别于完整 Monte Carlo 回报 `[0.81, 0.9, 1]`。读到这里，应当能分清“奖励”“实际回报”“回报目标”和“优势估计”四个对象。

再看边界处理的影响：假设另一个尚未结束的任务产生同样的三步奖励和前三个价值预测，但第四个状态预测为 2；只是采集预算用完， $\chi_3=0$ 。此时最后残差改为：

$$
\delta_3^{\mathrm{truncated}}=1+0.9\times2-0.6=2.2
$$

沿相同递推得到优势 `[1.40128, 1.724, 2.2]`。两种配置在片段末尾都停止累加，却因是否保留未来价值而得到不同目标。若任务真正结束，即使网络对最后状态输出 2，也应通过终止标记把这项 bootstrap 清零。

## 6. PPO 的裁剪策略目标

### 6.1 重要性采样：换一个分布，也能估计原期望

假设希望计算目标分布 $p$ 下函数 $f(z)$ 的期望，却只能从分布 $q$ 采样。在目标分布有概率的地方，采样分布也必须有非零概率；并且所求期望要存在。离散情形下，有如下恒等式：

$$
\mathbb{E}_{z\sim p}[f(z)]=\sum_zp(z)f(z)=\sum_{z:q(z)\gt0}q(z)\frac{p(z)}{q(z)}f(z)=\mathbb{E}_{z\sim q}\left[\frac{p(z)}{q(z)}f(z)\right]
$$

比值 $p(z)/q(z)$ 是重要性权重。连续分布时把求和换成积分，原理相同。它校正的是“这个值在目标分布与实际采样分布下出现的频率不同”。

用两个动作的例子手算：目标分布选 A、B 的概率为 `0.25、0.75`，采样分布的概率为 `0.5、0.5`，两个动作的函数值分别为 `2、0`。

| 动作 | 目标概率 p | 采样概率 q | 函数值 f | 重要性权重 p/q | 加权样本值 |
| --- | --- | --- | --- | --- | --- |
| A | 0.25 | 0.5 | 2 | 0.5 | 1 |
| B | 0.75 | 0.5 | 0 | 1.5 | 0 |

目标期望是 `0.25 × 2 + 0.75 × 0 = 0.5`。直接平均 q 分布下的函数值，会得到 `1`；平均加权样本值，则得到 `0.5 × 1 + 0.5 × 0 = 0.5`，恢复了目标期望。

恒等式正确，并不表示有限样本估计一定稳定。再看一个稀有大权重例子：目标分布选 B 的概率为 `0.5`，采样分布选 B 的概率只有 `0.01`，函数只在 B 上取 1，在 A 上取 0。大多数加权样本为 0；偶尔采到 B 时，样本值却是 `0.5 / 0.01 = 50`。在 100 次采样中，若没有采到 B，平均值是 0；采到一次是 0.5；采到两次是 1。真实期望仍是 0.5，但小批次会明显波动。

因此，采样分布与目标分布相差太远时，少数大权重可能支配估计；若采样分布根本不覆盖目标分布的某些动作，增加采样量也无法补救。PPO 要限制旧数据上的更新幅度，正与这些困难有关。

### 6.2 从策略梯度到新旧策略概率比

最直接的策略梯度根据采样动作的 log probability 和优势更新策略。但 PPO 希望对一批 rollout 做有限次 minibatch 更新：采样只发生一次，当前策略却持续变化，旧样本中的动作频率就不再等于当前策略的动作概率。

为处理这个差异，PPO 使用动作概率比衡量当前策略相对旧策略的变化：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\mathrm{old}}(a_t\mid s_t)\right)
$$

刚开始更新时，当前策略与旧策略相同，因此 $\rho_t(\theta)=1$ 。随着同一批数据被训练多个 minibatch epoch，二者逐渐产生差异。例如旧策略给已采样动作的概率是 0.2，新策略给出 0.3，那么概率比是 1.5，表示该动作在新策略下比采样时更常出现。

在重要性采样恒等式中，固定状态 $s_t$ 后，把目标分布替换为当前策略，把采样分布替换为旧策略，就得到这个动作概率比。未裁剪的代理目标是：

$$
\mathcal{L}_{\mathrm{surrogate}}(\theta)=\mathbb{E}_t\left[\rho_t(\theta)\hat A_t\right]
$$

在新旧策略相同的起点，对这个目标求梯度，会得到以旧策略样本和优势估计表达的策略梯度方向。这里仍使用旧策略采集的状态和旧策略下估计的优势；单步动作概率比没有把整个轨迹的状态分布也变成当前策略的分布，所以它是旧策略附近的**代理目标**，不是任意大幅更新后真实回报的精确表达式。

本文的 $\mathbb{E}_t$ 表示对有效采样位置的平均，主要对应有限 LLM 回答中常用的 $\gamma=1$ 情形。若一般 RL 任务希望精确对应 RL 基础理论第 7.3 节“从初始状态出发的折扣目标”，还需要相应的时间权重 $\gamma^{t-1}$ 或等价的折扣状态分布；不能把均匀位置平均自动当成那个目标的精确梯度。均匀位置平均与折扣目标梯度的区别，及其与 GAE 论文记号的关系，见附录 A.1。

还需核对变长回答的聚合方式。RL 基础理论第 7.6 节的 episode 目标对应“每条轨迹的 token 梯度求和，再对轨迹平均”。改为先对每条回答除以自己的长度，会改变样本权重；把全批 token 总和除以全批有效 token 数，在固定批次中只是整体缩放，但这个分母随采样变化，也不能自动声称其期望是原目标的无偏梯度。第 7 节伪代码应按选定的聚合方式实现，GRPO 基础理论第 9.4 节会进一步解释回答内平均的影响。

旧 log probability 必须保留采样时的值，优势与回报目标也应固定。如果每次更新都把分母改成当前模型，概率比会被错误地重置为 1，PPO 就失去了衡量策略变化的参照。

### 6.3 裁剪代理目标

PPO 的核心是最大化下面的裁剪代理目标：

$$
\mathcal{L}_{\mathrm{clip}}(\theta)=\mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat A_t,\,\mathrm{clip}\left(\rho_t(\theta),1-\varepsilon,1+\varepsilon\right)\hat A_t\right)\right]
$$

其中， $\varepsilon$ 是裁剪范围。这个目标可以分两种情况理解：

- 当 $\hat A_t\gt0$ 时，希望增大动作概率，但比例超过 $1+\varepsilon$ 后，不再从代理目标获得额外收益；
- 当 $\hat A_t\lt0$ 时，希望减小动作概率，但比例低于 $1-\varepsilon$ 后，不再因继续减小而获得额外收益。

对单个样本，也可以把这个目标改写成更容易看清方向的分段形式：

$$
\ell_t^{\mathrm{clip}}(\theta)=\begin{cases} \hat A_t\min\left(\rho_t(\theta),1+\varepsilon\right),&\hat A_t\ge0,\\ \hat A_t\max\left(\rho_t(\theta),1-\varepsilon\right),&\hat A_t\lt0. \end{cases}
$$

正优势只截住“概率增得过多”的收益；负优势只截住“概率减得过多”的收益。若更新方向对该样本不利，仍保留相应惩罚。取两个分支的较小值，是为了削弱“朝有利方向一次改得过头”的激励，不是证明真实回报每次更新都一定提高。

需要注意，clipping 裁剪的是代理目标的一个分支，**没有把模型概率或实际概率比强制限制在裁剪区间内**，也不保证更新后策略的真实 KL 一定低于某个阈值。未采样动作、共享参数引起的联动，以及多次梯度更新都可能让实际策略变化超出直觉范围。因此仍应监控新旧策略 KL、clip fraction 和概率比异常值，必要时提前停止当前批次的更新。

clip fraction 的常见统计口径是“有效采样位置中，概率比超出裁剪区间的比例”。另一些实现统计“min 实际采用裁剪分支的比例”；后者还与优势符号有关，二者应按框架实现核对，不能混用。

### 6.4 手算一次 clipping

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

### 6.5 总训练损失与 KL 放在哪里

将策略目标、价值损失和可选的熵奖励组合，可以写成：

$$
\mathcal{L}_{\mathrm{total}}(\theta,\psi)=-\mathcal{L}_{\mathrm{clip}}(\theta)+c_V\mathcal{L}_{V}(\psi)-c_H\mathcal{H}(\pi_\theta)
$$

其中， $c_V$ 和 $c_H$ 是权重， $\mathcal{H}(\pi_\theta)$ 是有效状态上的策略熵。最前面的负号是因为优化器通常最小化 loss，而策略目标需要最大化。LLM RLHF 实现不一定显式加入熵奖励，因为采样温度和参考模型 KL 已能影响探索；不能仅凭公式判断某个框架一定包含该项。

这个总损失沿用第 4.2 节的**奖励整形**配置：先把 KL 惩罚放进 token 奖励，再计算优势和回报目标，因此 KL 已通过这些学习信号进入 PPO 更新。另一类实现只用任务奖励构造优势，而直接在 actor loss 上加 KL 正则：

$$
\mathcal{L}_{\mathrm{total}}^{\mathrm{direct}}=-\mathcal{L}_{\mathrm{clip}}^{\mathrm{task}}+\beta\mathcal{K}(\theta)+c_V\mathcal{L}_{V}-c_H\mathcal{H}(\pi_\theta)
$$

这里的 $\mathcal{K}(\theta)$ 表示框架定义并估计的参考策略 KL 项。两种放置方式在采样估计、critic 目标和实际更新上不必完全等价，应明确记录采用哪一种。**不要在未明确设计的情况下，既用同一 KL 项整形奖励，又在 loss 中按相同系数再扣一次**。

在两种配置中，采样奖励、旧 log probability、旧价值、优势及回报目标都是固定数据；当前策略概率和当前价值预测必须保留计算图。actor 更新应阻止梯度通过优势权重传播，critic 更新也应阻止梯度通过回报目标传播。如果两个模型共享主干，两个损失仍可能共同改变共享参数，不能把“目标已冻结”误解成“critic 不会影响策略”。

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

开始第一轮采样前，旧策略与当前策略应同步。之后，每个 rollout 都携带自己的旧概率、旧价值、固定优势、固定回报目标和有效位置 mask，一起切分 minibatch。

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
            ratio, minibatch.advantages.detach(),
            response_mask=minibatch.response_mask
        )
        value_loss = clipped_value_loss(
            new_values, minibatch.returns.detach(), minibatch.old_values,
            response_mask=minibatch.response_mask
        )

        optimize(policy_loss + value_coefficient * value_loss)

    old_policy.load_state_dict(policy.state_dict())
```

这里的函数是框架无关的流程占位，并非可直接运行的 API。`.detach()` 显示了固定权重的语义；旧概率、旧价值等也由 `no_grad()` 下的采样步骤保存。`bootstrap_values` 对真正终止的回答应为 0，对仅被采集预算截断的片段应取最后动作之后状态的价值。伪代码沿用“KL 放进奖励”的配置，展示了 response mask 和终止信息的传递，省略了其内部实现、attention mask、分布式通信、梯度累积和数值稳定处理。真实实现中，所有逐 token 张量必须只在有效回答位置参与计算。

不要把更新策略和价值模型的前向过程也包进 `no_grad()`：那样当前 log probability 和当前 value 就无法对待训练参数反向传播。相反，也不能让 actor 的梯度穿过优势计算去改变 critic，以“修改评价权重”代替“改进动作概率”。

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

原始 GRPO 采用“旧策略采样、概率比、clipping、参考模型 KL”等结构，却用组内相对奖励替代学习得到的价值函数。其中，概率比和 clipping 对应 PPO 式策略更新，参考模型 KL 对应常见 LLM-PPO 配方中的长期正则，并非 PPO 算法本身必须具备的组件。GRPO 减少了显存和训练复杂度，但会引入组内采样成本以及新的方差和偏差问题，详见 [GRPO 基础理论](./03_grpo基础理论.md)。

## 13. 本章小结

- LLM 生成可以看作逐 token 决策过程，完整回答是轨迹，奖励评价生成结果。
- 本章的典型 actor-critic 配方使用价值模型和 GAE；PPO-Clip 的裁剪目标本身不规定优势必须来自 critic。
- 旧策略用于约束一次 rollout 上的更新，参考策略用于限制相对初始模型的长期漂移，二者不能混淆。
- KL 约束、PPO clipping、奖励尺度、response mask 和 on-policy 数据共同影响训练稳定性。
- PPO 能直接优化不可微的序列级目标，但需要维护多个模型、在线生成数据，并谨慎防范 reward hacking。

## 14. 数值核验与学完后自检

### 14.1 用普通 CPU 核对公式

[数值核验脚本](./examples/verify_rl_math.py) 用 Python 3 标准库核对基础回报、动作无关基线、重要性采样、GAE 与 lambda-return 等关系，不执行模型训练，不需要 GPU 或外部数据，也不创建模型输出目录。在仓库根目录运行：

```bash
python3 rl/examples/verify_rl_math.py
```

先独立手算第 5.8 节和第 6.1、6.4 节，再用脚本核对。数值核验帮助发现公式、权重和边界实现中的错误，不能替代真实任务中的训练与独立评估。

### 14.2 进入 GRPO 之前，能否解释这些问题

1. **旧策略和参考策略分别在什么时候同步？** 旧策略通常随下一轮采样刷新；参考策略通常整个阶段冻结。它们分别服务于本轮概率比和长期 KL 锚点。
2. **为什么要保留采样时的 log probability？** 它描述产生数据的策略。如果每次更新都用当前策略覆盖分母，就失去了衡量变化的参照。
3. **一次高奖励就定义了真实正优势吗？** 不是；真实优势比较条件期望，单条轨迹只提供含噪声的估计。
4. **有限 lambda-return 最后一项为什么没有乘“1 减 lambda”？** 因为它承接所有更长跨度的剩余权重，保证有限片段的权重之和为 1。
5. **lambda 为 1 时，critic 不准是否一定让梯度有偏？** 完整终止回报减固定动作无关基线，不改变梯度期望；截断片段的尾部 bootstrap 误差则不能这样消去。
6. **触及采样长度上限是否总把未来价值清零？** 不是；要看任务真正结束还是仍可继续。GAE 停止累加与 bootstrap 是否为 0 是不同的边界判断。
7. **正优势动作的概率比从 1.2 增到 1.5，裁剪目标还能增加吗？** 当裁剪上界为 1.2 时不能；但实际概率比仍可以达到 1.5，clipping 没有投影模型概率。
8. **概率比校正了整条轨迹的新状态分布吗？** 没有；这里校正的是旧采样状态上的动作概率，代理目标仍使用旧状态与旧优势。
9. **stop-gradient 是否表示 actor 和 critic 都不能求导？** 不是；冻结的是旧数据与目标，当前概率和价值预测仍需要计算图。
10. **只有终局奖励，前面的动作为什么仍能学？** 回报、基线和优势估计把未来结果传回各位置；这提供学习信号，但不等于精确识别了每个 token 的因果贡献。

能手算并解释这些问题后，进入 [GRPO 基础理论](./03_grpo基础理论.md)。附录 A 的形式化条件可继续按需复习。

## 附录 A：gamma-just 究竟保证什么

本附录对应 GAE 原论文第 2 节的定义 1、命题 1，第 3 节的推导，以及第 6.1 节关于旧 critic 的说明。首次阅读可先掌握正文中的手算和训练流程，再回来区分“优势估计”与“梯度估计”的无偏性。

### A.1 先区分两个带 gamma 的对象

为与 GAE 论文对照，这里把折扣价值写成 $V^{\pi,\gamma}$ 、 $Q^{\pi,\gamma}$ 和 $A^{\pi,\gamma}$ 。它们与前文固定折扣因子后的 $V^\pi$ 、 $Q^\pi$ 和 $A^\pi$ 表示同一组概念。

基础章将折扣直接写入任务目标，从初始状态出发最大化：

$$
J_\gamma(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=1}^{T}\gamma^{t-1}r_t\right]
$$

在环境与奖励不直接依赖策略参数的条件下，对这个目标求精确策略梯度，需要保留外层时间权重：

$$
\nabla_\theta J_\gamma(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=1}^{T}\gamma^{t-1}A^{\pi_\theta,\gamma}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
$$

GAE 原论文首先讨论的是**未折扣总奖励目标**，再把 $\gamma$ 引入为估计折中参数。论文公式（6）定义的方向是：

$$
g^\gamma=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=1}^{T}A^{\pi_\theta,\gamma}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
$$

这里按本仓库改为从 1 开始的有限轨迹记号；原论文写成从 0 开始的无限求和。注意， $g^\gamma$ **没有外层时间权重**，因此一般不能把它直接等同于上面的精确折扣目标梯度。原论文是在原来的未折扣问题中讨论这一近似所带来的偏差；若任务本来定义为折扣目标，则应按该目标的权重分析。有限 LLM 回答常取 $\gamma=1$ ，此时这个区别消失。

### A.2 梯度期望正确，不要求优势数值无偏

定义 1 中， $\gamma$ -just 指：用优势估计替换真实折扣优势后，每个位置的**策略梯度项期望**保持一致：

$$
\mathbb{E}_{\tau\sim\pi_\theta}\left[\hat A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]=\mathbb{E}_{\tau\sim\pi_\theta}\left[A^{\pi_\theta,\gamma}(s_t,a_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]
$$

它不是要求每次采样都得到真实优势，也不是直接要求以下数值条件：

$$
\mathbb{E}[\hat A_t\mid s_t,a_t]=A^{\pi_\theta,\gamma}(s_t,a_t)
$$

优势数值的条件无偏可以保证梯度项期望一致，但梯度项期望一致不必要求优势数值条件无偏。GAE 原论文命题 1 给出的一个充分条件是：估计量可写成无偏的折扣动作价值估计减去一个在当前动作采样前已确定的基线。例如：

$$
\hat A_t=\hat Q_t-b_t,\qquad\mathbb{E}[\hat Q_t\mid s_t,a_t]=Q^{\pi_\theta,\gamma}(s_t,a_t)
$$

其中， $b_t$ 可以依赖当前状态和此前的历史，但不能随当前采样动作变化。状态基线的梯度期望为 0：

$$
\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}\left[b(s_t)\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right]=b(s_t)\nabla_\theta\sum_a\pi_\theta(a\mid s_t)=0
$$

所以基线不必数值准确，仍可不改变梯度期望。条件是基线对当前动作无关，并且 actor 计算时不对它求导；基线越不合适，方差却可能越大。不能把这条结论改写成“任何 baseline 都降低方差”或“critic 不准也总是无害”。

### A.3 完整回报与截断 bootstrap 的区别

在真正终止的完整 episode 上， $\lambda=1$ 给出：

$$
\hat A_t=R_t-V_{\mathrm{old}}(s_t)
$$

完整回报在给定状态和动作后的期望是相应动作价值。因此，即使旧 critic 预测不准，固定的状态基线仍满足附录 A.2 中的 $\gamma$ -just 梯度条件。但其优势数值的条件期望为：

$$
\mathbb{E}[\hat A_t\mid s_t,a_t]=Q^{\pi,\gamma}(s_t,a_t)-V_{\mathrm{old}}(s_t)
$$

只有当旧价值恰好等于真实状态价值时，这个表达式才等于真实优势。

用一个一步任务核对：A、B 的概率为 `0.6、0.4`，确定性奖励为 `2、0`。真实状态价值为 1.2，真实优势为 `0.8、-1.2`；若旧 critic 错误地预测价值为 10，完整回报减基线得到 `-8、-10`，显然不是那两个真实优势。以动作 A 在 softmax 转换前的实数得分（logit）为参数，对数概率梯度在 A、B 上分别为 `0.4、-0.6`，两种估计的梯度期望却相同：

$$
0.6\times0.8\times0.4+0.4\times(-1.2)\times(-0.6)=0.48
$$

$$
0.6\times(-8)\times0.4+0.4\times(-10)\times(-0.6)=0.48
$$

这个例子说明：优势数值有偏，不等于策略梯度期望必定有偏。

如果只收集了一段尚未终止的轨迹， $\lambda=1$ 仍保留尾部预测：

$$
\hat A_t=\sum_{l=0}^{K-1}\gamma^lr_{t+l}+\gamma^KV_{\mathrm{old}}(s_{T+1})-V_{\mathrm{old}}(s_t)
$$

尾部状态由当前动作及后续轨迹影响，尾部价值误差不能简单当成当前状态的动作无关基线消去。因此， $\lambda=1$ 在这种截断配置下**不自动保证**梯度条件；正确的尾部价值是一个充分条件。一步 TD 在使用真实折扣价值时，也可保证优势数值条件无偏；价值预测不准时一般会引入 bootstrap 偏差。

这些结论都在指定采样策略下讨论。旧 critic 应在本批采样前确定，优势计算后保持固定；若先拟合本批数据再重算 actor 的基线，预测可能混入当前动作的采样信息，不能直接沿用动作无关基线的论证。GAE 原论文第 6.1 节也明确使用策略更新前的价值模型来估计优势。

最后， $\gamma$ -just 讨论的是用估计优势构造**未裁剪策略梯度**时的期望条件。它不证明 PPO clipping 后或反复更新旧数据时的梯度始终等于真实目标梯度，也不保证每次更新都提高真实回报。

## 参考文献

1. Schulman, J., et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
2. Schulman, J., et al. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR, 2016.
3. Ziegler, D. M., et al. [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593). 2019.
4. Stiennon, N., et al. [Learning to Summarize with Human Feedback](https://arxiv.org/abs/2009.01325). NeurIPS, 2020.
5. Ouyang, L., et al. [Training Language Models to Follow Instructions with Human Feedback](https://arxiv.org/abs/2203.02155). NeurIPS, 2022.
6. Bai, Y., et al. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862). 2022.
