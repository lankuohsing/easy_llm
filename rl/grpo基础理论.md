# GRPO（Group Relative Policy Optimization）基础理论

监督微调（SFT）通过模仿参考答案提高模型生成这些答案的概率，但很多任务只有最终结果容易判断，很难为每个问题编写完整且唯一的标准推理过程。强化学习允许模型先从当前策略中采样回答，再根据奖励提高较好回答的生成概率、降低较差回答的生成概率。GRPO（Group Relative Policy Optimization，组相对策略优化）的核心做法是：对同一个提示词采样一组回答，用组内相对奖励估计优势，不需要训练独立的价值模型（critic）。

GRPO 保留了 [PPO](./ppo基础理论.md) 风格的旧策略概率比、clipping 和参考模型 KL 等结构，但本章不把 PPO 作为阅读前提。下面会从语言生成的强化学习建模开始，完整定义所需符号、训练角色、目标函数和训练流程，再讨论二者的异同。

## 1. GRPO 从哪里来

GRPO 由 DeepSeekMath 工作系统提出，用于提升语言模型的数学推理能力，后来也被 DeepSeek-R1 等推理模型训练工作采用。它适合下面这类场景：

- 同一个 prompt 可以采样多个候选回答；
- 每个回答能够由规则、验证器或奖励模型打分；
- 回答之间的相对好坏比绝对价值更容易判断；
- 训练资源不适合再维护一个大型 critic。

典型流程可以概括为：

```text
一个 prompt
  ↓ 从旧策略采样 G 个回答
回答 1、回答 2、...、回答 G
  ↓ 分别计算奖励
r₁、r₂、...、r_G
  ↓ 组内中心化与标准化
相对优势 A₁、A₂、...、A_G
  ↓ clipped policy objective + 参考模型 KL
更新策略模型
```

GRPO 不是与 PPO 完全无关的新范式。它保留了旧策略概率比、clipping 和参考模型 KL 等核心结构，主要改变的是**优势如何估计**：PPO 使用学习得到的价值函数作为基线，GRPO 使用同一 prompt 下的组内奖励统计量作为基线。

## 2. 把语言生成写成 GRPO 问题

### 2.1 状态、动作、策略与轨迹

设训练提示词来自数据集或提示词分布 $\mathcal{D}$ 。给定一条提示词 $x$ ，语言模型逐 token 生成回答 $y=(y_1,\ldots,y_T)$ ，其中 $T$ 是该回答的有效 token 数量。这个过程可以写成有限时长的序列决策问题：

| 强化学习概念 | 在语言模型中的含义 |
| --- | --- |
| 初始条件 | 提示词 $x$ |
| 状态 $s_t$ | 提示词和已经生成的回答前缀，即 $s_t=(x,y_{\lt t})$ |
| 动作 $a_t$ | 在位置 $t$ 选择的下一个 token，即 $a_t=y_t$ |
| 策略 $\pi_\theta(a_t\mid s_t)$ | 参数为 $\theta$ 的语言模型给出的 next-token 概率 |
| 轨迹 $\tau$ | 从提示词开始，直到结束 token 或长度上限的完整生成过程 |
| 奖励 $r(x,y)$ | 对完整回答或中间推理步骤质量的数值评价 |

策略（policy）就是模型在每个状态下对下一个 token 给出的概率分布。根据自回归分解，完整回答的生成概率为：

$$
\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{\lt t})
$$

一次完整生成通常称为一个 rollout。GRPO 不要求模型复现某个固定参考答案，而是从策略分布中实际采样回答，再用奖励判断这条轨迹是否值得强化。若回答的优势为正，训练会倾向于提高其中已采样 token 的概率；若优势为负，训练会倾向于降低这些 token 的概率。

### 2.2 组采样

GRPO 不只为提示词 $x$ 生成一个回答，而是使用旧策略独立采样 $G$ 个候选回答：

$$
\{y_i\}_{i=1}^{G}\sim\pi_{\mathrm{old}}(\cdot\mid x)
$$

其中，第 $i$ 个回答写成 $y_i=(y_{i,1},\ldots,y_{i,T_i})$ 。它在 token 位置 $t$ 的状态和动作分别是：

$$
s_{i,t}=(x,y_{i,\lt t}),\qquad a_{i,t}=y_{i,t}
$$

同一个 prompt 下的 $G$ 个回答构成一个组。GRPO 使用这组回答的奖励均值作为局部基线，判断某个回答相对同组平均水平更好还是更差。

### 2.3 符号约定

本文使用的核心符号如下。后续公式中的符号均以此表为准：

| 符号 | 含义 |
| --- | --- |
| $\mathcal{D}$ | 训练提示词的数据集或分布 |
| $\mathcal{V}$ | 模型 tokenizer 的词表，即每一步可选择的 token 集合 |
| $x$ | 当前提示词，可包含 system 消息、历史对话和用户问题 |
| $\tau$ | 一次完整生成轨迹，包含提示词、逐 token 动作和结束状态 |
| $G$ | 每个提示词采样的回答数量，也称 group size |
| $i$ | 组内回答下标， $i\in\{1,\ldots,G\}$ |
| $j$ | 计算组奖励统计量时使用的求和下标，同样遍历组内 $G$ 个回答 |
| $y_i$ | 第 $i$ 个完整回答，即一条采样轨迹的回答部分 |
| $T_i$ | 回答 $y_i$ 的有效 token 数量，不包含 padding |
| $t$ | 回答内的 token 位置下标， $t\in\{1,\ldots,T_i\}$ |
| $y_{i,t}$ | 回答 $y_i$ 中的第 $t$ 个 token |
| $y_{i,\lt t}$ | 回答 $y_i$ 中位于位置 $t$ 之前的 token 前缀 |
| $s_{i,t}$ | 生成第 $i$ 个回答的第 $t$ 个 token 前的状态，即 $(x,y_{i,\lt t})$ |
| $a_{i,t}$ | 状态 $s_{i,t}$ 下实际采样的动作，即 token $y_{i,t}$ |
| $\theta$ | 当前策略中参与训练的参数 |
| $\pi_\theta$ | 当前策略，即正在接受梯度更新的语言模型 |
| $\pi_{\mathrm{old}}$ | 生成当前批次回答的旧策略快照，在这批数据的更新期间保持冻结 |
| $\pi_{\mathrm{ref}}$ | 冻结的参考策略，用于计算 KL 惩罚并约束长期行为漂移 |
| $r(x,y_i)$ 或 $r_i$ | 奖励函数为第 $i$ 个完整回答给出的序列级奖励 |
| $\bar r$ | 同一提示词下 $G$ 个回答的平均奖励 |
| $\sigma_r$ | 同一提示词下组奖励的标准差 |
| $\epsilon_{\mathrm{std}}$ | 奖励标准化分母中的数值稳定常数 |
| $\hat A_i$ | 第 $i$ 个回答的组相对优势估计 |
| $\hat A_{i,t}$ | 分配到回答 $i$ 的 token 位置 $t$ 的优势估计 |
| $\rho_{i,t}(\theta)$ | 当前策略与旧策略对已采样 token 给出的概率之比 |
| $\varepsilon$ | 策略概率比的裁剪超参数 |
| $\beta$ | 参考模型 KL 惩罚的权重 |
| $d_{i,t}$ | 在状态 $s_{i,t}$ 上，对当前策略与参考策略之间正向 KL 散度的逐 token 采样估计项 |
| $\mathcal{J}_{\mathrm{GRPO}}(\theta)$ | 需要最大化的 GRPO 策略目标 |
| $V_\psi(s_t)$ | 仅在与 PPO 对比时出现，表示 PPO 的 critic 对状态未来回报的预测；GRPO 不训练该模型 |

这里的“有效 token”包括实际回答内容以及训练实现选择保留的结束 token，但不包括为组成 batch 而额外添加的 padding。prompt token 为生成提供条件，通常也不进入策略损失。

### 2.4 训练角色及其关系

GRPO 中常见的训练角色如下：

| 角色 | 常用符号 | 是否更新 | 作用 |
| --- | --- | --- | --- |
| 当前策略 | $\pi_\theta$ | 是 | 计算新概率并接受梯度更新，也是最终要得到的模型 |
| 旧策略 | $\pi_{\mathrm{old}}$ | 一轮更新期间冻结 | 生成当前组回答，并作为策略概率比的分母 |
| 参考策略 | $\pi_{\mathrm{ref}}$ | 否 | 提供 KL 约束，抑制策略长期漂移 |
| 奖励函数 | $r(x,y)$ | GRPO 阶段通常固定 | 为每个完整回答或中间步骤打分 |
| 价值模型 | 无 | 不需要 | GRPO 使用组内奖励基线，不训练 critic |

当前策略通常由 SFT 模型初始化。训练开始时，可以复制该模型得到冻结的参考策略；每轮 rollout 时，再把采样所用策略视为旧策略。之后当前策略会在同一批回答上做若干次梯度更新，而旧策略在这期间不变。进入下一轮 rollout 时，再使用更新后的策略重新采样。

旧策略与参考策略不能混淆：

- $\pi_{\mathrm{old}}$ 描述“当前批回答由哪个策略生成”，会随训练轮次更新；
- $\pi_{\mathrm{ref}}$ 描述“训练后的策略不应偏离哪个行为锚点太远”，通常在整个 GRPO 阶段保持冻结。

奖励函数 $r(x,y)$ 可以是神经网络奖励模型，也可以是答案校验器、单元测试、格式检查器或多个信号的加权组合。**GRPO 不需要 critic，不等于 GRPO 不需要奖励。** DeepSeek-R1-Zero 在可验证推理任务中主要使用准确性奖励和格式奖励，但 GRPO 本身并不限定奖励来源。

## 3. 组相对优势是如何得到的

在策略优化中，奖励说明一条完整回答得分多少，优势（advantage）则说明这个回答或动作相对某个基线好多少。训练时，正优势用于提高已采样动作的概率，负优势用于降低其概率。如果直接把原始奖励当作更新权重，不同 prompt 的难度和奖励尺度会造成很大方差：一个困难问题的最佳回答可能仍然只有低分，一个简单问题的普通回答却可能得到高分。

基线（baseline）的作用是提供一个可比较的预期水平。减去不依赖当前动作的合适基线，可以降低策略梯度方差，而不改变理想情况下的优化方向。PPO 通常训练价值模型预测状态基线；GRPO 不训练价值模型，而是把同一个 prompt 下多个采样回答的平均奖励当作经验基线。这是“Group Relative”名称的来源。

奖励函数本身不必可微。答案比较器、代码执行结果或人工评分都可以只输出标量奖励；策略更新时把奖励和由它计算出的优势视为固定权重，梯度通过当前策略的 token 概率传播，而不需要穿过奖励函数。

### 3.1 对同一 prompt 进行组采样

对提示词 $x$ 采样 $G$ 个回答后，分别计算序列级奖励：

$$
r_i=r(x,y_i),\qquad i\in\{1,\ldots,G\}
$$

“同一 prompt”是关键条件。不同问题的难度和奖励标尺可能完全不同，不能随意把一批无关 prompt 的回答混在一起计算组均值。

组奖励均值为：

$$
\bar r=\frac{1}{G}\sum_{j=1}^{G}r_j
$$

组奖励标准差为：

$$
\sigma_r=\sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j-\bar r)^2}
$$

最常见的序列级组相对优势写成：

$$
\hat A_i=\frac{r_i-\bar r}{\sigma_r+\epsilon_{\mathrm{std}}}
$$

其中， $\epsilon_{\mathrm{std}}$ 是防止除零的小常数。不同实现可能采用样本标准差、总体标准差或不除以标准差，公式细节会影响奖励尺度，复现实验时必须检查具体代码。

减去组均值后：

- 奖励高于同组平均值的回答得到正优势，策略会倾向于提高其 token 概率；
- 奖励低于同组平均值的回答得到负优势，策略会倾向于降低其 token 概率；
- 绝对奖励不重要，组内排序和差距决定本次更新方向。

### 3.2 一个简单例子

假设同一个数学问题采样 4 个回答，奖励分别为：

```text
[1, 1, 0, 0]
```

组均值为 $0.5$ ，总体标准差为 $0.5$ 。忽略数值稳定常数后，标准化优势为：

```text
[1, 1, -1, -1]
```

两个正确回答中的 token 会得到正向更新，两个错误回答中的 token 会得到负向更新。GRPO 不需要 critic 预测“这个问题通常能得多少分”，因为同组的其他回答已经提供了局部比较基线。

### 3.3 为什么所有奖励相同时没有学习信号

如果同组回答全部正确或全部错误，则每个 $r_i-\bar r$ 都等于 0，所有优势也都等于 0。即使分母加入了 $\epsilon_{\mathrm{std}}$ ，也只能避免数值除零，不能凭空产生策略梯度。

这会带来一个重要实践问题：

- 太容易的问题可能组内全对；
- 太难的问题可能组内全错；
- 只有能在当前策略下产生奖励差异的问题，才提供有效的相对学习信号。

因此，prompt 难度分布、每组样本数、采样温度和奖励分辨率都会直接影响有效训练样本比例。有些后续方法会动态过滤零方差组或调整采样策略，但这属于对基础 GRPO 的工程改进。

## 4. 从序列优势到 token 更新

使用结果奖励（outcome reward）时，一个回答通常只有一个标量优势 $\hat A_i$ 。最直接的做法是把它赋给该回答的所有有效 token：

$$
\hat A_{i,t}=\hat A_i,\qquad t\in\{1,\ldots,T_i\}
$$

把最终结果转化为各个 token 的学习信号，称为信用分配（credit assignment）。上式意味着只要最终答案相对同组更好，整条生成路径都会受到正向强化；只要最终答案相对同组更差，整条路径都会受到负向强化。优点是实现简单，不需要逐步标注；缺点是信用分配粗糙，正确答案中的无效绕路和错误答案中的合理步骤也会得到相同符号的信号。

若奖励函数能够对中间推理步骤打分，可以构造过程奖励（process reward），再根据当前位置之后的步骤奖励计算 token 或步骤级优势。DeepSeekMath 讨论了 outcome supervision 和 process supervision 两种形式。无论采用哪一种，都必须保证步骤边界、奖励位置和 response mask 对齐。

## 5. GRPO 的裁剪目标

### 5.1 新旧策略概率比

对于第 $i$ 个回答的第 $t$ 个 token，状态为 $s_{i,t}=(x,y_{i,\lt t})$ ，动作是 $a_{i,t}=y_{i,t}$ 。新旧策略概率比为：

$$
\rho_{i,t}(\theta)=\frac{\pi_\theta(a_{i,t}\mid s_{i,t})}{\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})}
$$

实际代码通常保存 log probability，再通过差值计算概率比：

$$
\rho_{i,t}(\theta)=\exp\left(\log\pi_\theta(a_{i,t}\mid s_{i,t})-\log\pi_{\mathrm{old}}(a_{i,t}\mid s_{i,t})\right)
$$

其中， $\log$ 表示自然对数， $\exp$ 表示指数函数。使用 log probability 能避免多个很小概率直接相乘造成数值下溢，也便于计算新旧策略的概率比。

概率比衡量当前策略对已采样动作的偏好相对采样时发生了多大变化：

- $\rho_{i,t}=1$ 表示当前策略与旧策略对该 token 给出的概率相同；
- $\rho_{i,t}\gt1$ 表示当前策略提高了该 token 的概率；
- $\rho_{i,t}\lt1$ 表示当前策略降低了该 token 的概率。

rollout 刚生成时，当前策略与旧策略通常相同，因此概率比接近 1。对同一批回答进行多次梯度更新后，当前策略逐渐偏离旧策略。使用这个比值，可以在回答来自旧策略的情况下，估计当前策略应如何更新；但偏离过大时估计会变得不可靠，因此还需要 clipping。

### 5.2 基础 GRPO 目标

裁剪函数把标量 $z$ 限制在下界 $l$ 和上界 $u$ 之间：

$$
\mathrm{clip}(z,l,u)=\min\left(\max(z,l),u\right)
$$

使用序列内 token 平均时，GRPO 的最大化目标可以写成：

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}\left(\min\left(\rho_{i,t}\hat A_{i,t},\,\mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)\hat A_{i,t}\right)-\beta d_{i,t}\right)\right]
$$

其中：

- $G$ 是每个 prompt 的回答数量；
- $T_i$ 是第 $i$ 个回答的有效 token 数量；
- $\varepsilon$ 是 PPO 风格的裁剪范围；
- $\beta$ 是参考模型 KL 惩罚系数，用于控制模型受到多强的行为约束；
- $d_{i,t}$ 是在状态 $s_{i,t}$ 上，对当前策略 $\pi_\theta$ 与参考策略 $\pi_{\mathrm{ref}}$ 之间正向 KL 散度的逐 token 采样估计；它衡量当前位置的策略分布偏离参考分布的程度；
- 最外层 $\mathbb{E}$ 表示对训练 prompt 和由旧策略采样的回答组取期望；实际训练用当前 minibatch 的均值进行估计。

因此，目标中的 $-\beta d_{i,t}$ 是 KL 惩罚项：偏离参考策略越明显，策略目标中被扣除的值越大。这里要区分两组比较关系： $\rho_{i,t}$ 比较当前策略与旧策略，用于限制这一轮更新； $d_{i,t}$ 比较当前策略与参考策略，用于限制整个训练过程的长期漂移。 $d_{i,t}$ 的具体计算公式和“采样估计”的含义将在 5.3 节展开。

先忽略 KL 项，目标中的乘积 $\rho_{i,t}\hat A_{i,t}$ 决定更新方向：优势为正时，增大该动作的概率比可以提高目标；优势为负时，减小该动作的概率比可以提高目标。`min` 与 `clip` 共同限制这种改动带来的收益：

- 当 $\hat A_{i,t}\gt0$ 时，希望提高动作概率，但 $\rho_{i,t}$ 超过 $1+\varepsilon$ 后，不再从代理目标获得额外收益；
- 当 $\hat A_{i,t}\lt0$ 时，希望降低动作概率，但 $\rho_{i,t}$ 低于 $1-\varepsilon$ 后，不再因继续降低而获得额外收益。

因此，clipping 鼓励策略进行相对保守的更新。它不是硬约束，也不能保证更新后的真实 KL 一定低于某个阈值；共享模型参数还会同时改变许多未采样 token 的概率。实践中仍需监控新旧策略 KL、clip fraction 和概率比极值。

这一部分与 PPO clipped objective 的形式相同，区别在于优势来自组内相对奖励，而不是价值模型和 GAE。 $1/T_i$ 使每条回答先在自身 token 上取平均，再在组内取平均；有些实现改为对整个 batch 的有效 token 统一平均，这会改变不同长度回答的相对权重。

上式写成需要最大化的策略目标。训练框架中的优化器通常最小化 loss，因此代码一般最小化 $-\mathcal{J}_{\mathrm{GRPO}}(\theta)$ 或与它等价的 minibatch 估计；伪代码中的负号正是来自这一步转换。

### 5.3 为什么需要参考模型与 KL 惩罚

如果只最大化任务奖励，策略可能找到奖励函数的漏洞。例如，模型可能反复输出某个容易触发格式奖励的模板，或者生成奖励模型偏爱但人类并不认可的文本。这称为 reward hacking。策略也可能迅速偏离原有语言分布，导致表达退化、重复或通用能力下降。

GRPO 使用冻结参考策略 $\pi_{\mathrm{ref}}$ 作为行为锚点，并惩罚当前策略与参考策略之间的 KL 散度。对状态 $s$ 上的完整词表 $\mathcal{V}$ ，正向 KL 可以写成：

$$
D_{\mathrm{KL}}\left(\pi_\theta\|\pi_{\mathrm{ref}}\right)=\sum_{a\in\mathcal{V}}\pi_\theta(a\mid s)\log\frac{\pi_\theta(a\mid s)}{\pi_{\mathrm{ref}}(a\mid s)}
$$

其中， $\mathcal{V}$ 是词表， $a$ 表示任意候选 token。两个分布完全相同时 KL 为 0，当前策略越偏离参考策略，KL 通常越大。GRPO 目标中的 $\beta$ 控制这种约束的强弱：

- $\beta$ 太小，策略更容易偏离参考模型并利用奖励漏洞；
- $\beta$ 太大，策略几乎不敢改变，任务奖励难以上升。

直接遍历完整词表计算每个位置的 KL 成本较高。原始 GRPO 公式使用下面的逐 token KL 估计形式：

$$
d_{i,t}=\frac{\pi_{\mathrm{ref}}(a_{i,t}\mid s_{i,t})}{\pi_\theta(a_{i,t}\mid s_{i,t})}-\log\frac{\pi_{\mathrm{ref}}(a_{i,t}\mid s_{i,t})}{\pi_\theta(a_{i,t}\mid s_{i,t})}-1
$$

对任意正数 $u$ ，都有 $u-\log u-1\ge0$ ，所以该形式对单个采样 token 也非负。当动作由当前策略分布采样并取期望时，它对应上面的正向 KL。将 $\beta d_{i,t}$ 从策略收益中减去，就会惩罚偏离参考模型的更新。

参考模型 KL 与 clipping 解决的问题不同：clipping 限制当前策略相对本轮旧策略的一次更新幅度，参考模型 KL 限制整个训练过程相对初始行为锚点的长期漂移。两者不能互相替代。

不同训练框架可能使用 sampled log-ratio、直接 KL、上述估计器，或把 KL 作为 token 奖励而不是显式目标项。它们的梯度和日志数值并不完全相同。阅读配置时不能只看到参数名 `beta` 或 `kl_coef` 就假设实现与原论文公式一致。

## 6. 一轮 GRPO 训练发生了什么

```text
从数据集中采样一批 prompt
  ↓
旧策略为每个 prompt 生成 G 个回答
  ↓
奖励函数分别评价每个回答
  ↓
对每个 prompt 单独计算组均值、标准差和相对优势
  ↓
当前策略计算 token log probability
参考策略计算 KL 所需概率
  ↓
使用 response mask 计算 clipped objective 与 KL
  ↓
更新策略若干次，丢弃旧 rollout，再重新采样
```

框架无关的伪代码如下：

```python
for prompts in prompt_dataloader:
    with no_grad():
        grouped_responses = old_policy.generate(
            prompts,
            num_return_sequences=group_size,
        )
        old_log_probs = old_policy.log_probs(prompts, grouped_responses)
        rewards = reward_fn(prompts, grouped_responses)
        advantages = normalize_within_each_prompt(rewards)

    for minibatch in repeat_and_shuffle(grouped_responses):
        new_log_probs = policy.log_probs(minibatch)
        ref_log_probs = reference_policy.log_probs(minibatch)

        ratio = (new_log_probs - minibatch.old_log_probs).exp()
        policy_objective = clipped_objective(
            ratio=ratio,
            advantages=minibatch.advantages,
            response_mask=minibatch.response_mask,
        )
        kl_penalty = compute_kl_penalty(
            new_log_probs=new_log_probs,
            ref_log_probs=ref_log_probs,
            response_mask=minibatch.response_mask,
        )

        optimize(-policy_objective + kl_coefficient * kl_penalty)

    old_policy.load_state_dict(policy.state_dict())
```

实际实现还要处理每个 prompt 的分组索引、不同长度回答、分布式全局统计、截断回答、重复样本和数值稳定性。若打乱 minibatch 后丢失 prompt 到回答的分组关系，计算出的就不再是组相对优势。

## 7. GRPO 与 PPO 的对比

为了独立理解下表，先简要说明 PPO 的基线：PPO 训练一个价值模型 $V_\psi(s_t)$ ，预测从状态 $s_t$ 继续生成时能够获得的未来累计奖励。它通常再用广义优势估计（Generalized Advantage Estimation，GAE）组合逐步奖励和价值预测，为不同 token 位置得到不同的优势。GRPO 删除了这个学习得到的价值模型，改用同一 prompt 的组内平均奖励作为基线；在结果监督下，同一回答的所有有效 token 通常共享一个优势。

| 对比维度 | PPO | GRPO |
| --- | --- | --- |
| 优势基线 | 价值模型 $V_\psi(s_t)$ | 同一 prompt 的组奖励均值 |
| 优势估计 | 常用 GAE，能够随 token 变化 | 结果监督下同一回答的 token 常共享优势 |
| 是否需要 critic | 需要 | 不需要 |
| 每个 prompt 的采样 | 可以只采一个或少量回答 | 必须形成一组回答才有相对基线 |
| 主要额外成本 | 价值模型训练与存储 | 多回答 rollout 与奖励计算 |
| 对奖励尺度的处理 | 常做 batch 级归一化或 whitening | 常做 prompt 组内中心化和标准化 |
| 容易遇到的问题 | value loss 不稳、critic 偏差 | 零方差组、组大小敏感、长度和难度偏差 |
| 共同结构 | 旧策略采样、概率比、clipping、参考模型 KL、on-policy 更新 | 旧策略采样、概率比、clipping、参考模型 KL、on-policy 更新 |

“GRPO 比 PPO 更省资源”主要指省掉 critic，并不表示总计算量一定更小。若每个 prompt 需要生成很多长回答，rollout 和验证器的成本仍然很高。反过来，PPO 的 critic 能对单条轨迹的每个状态提供不同基线，在组采样昂贵或奖励极稀疏时可能更有价值。

## 8. GRPO 为什么适合可验证推理任务

数学、代码和部分结构化任务具有较可靠的自动验证信号：最终答案可以做等价性检查，代码可以运行测试，格式可以用解析器验证。对同一个题目采样多条推理路径后，GRPO 能直接提高成功路径的相对概率，而不必为每个 token 编写参考答案。

这种训练方式可能让模型通过探索发现 SFT 数据中没有出现过的解题路径。它尤其适合“答案容易验证，但完整高质量推理过程昂贵”的任务。

不过，可验证不等于奖励天然可靠：

- 字符串匹配可能把等价答案判错，也可能接受偶然命中的错误推理；
- 单元测试覆盖不足时，代码可能通过测试却不满足真实需求；
- 格式奖励权重过大时，模型可能只学会模板而没有提升能力；
- 若训练题目或验证器泄漏，奖励上升不代表泛化改善。

因此，规则奖励本身也需要测试、对抗样例和人工抽查。

## 9. 实践中的关键细节

### 9.1 组必须按 prompt 正确构造

组均值只能在同一个 prompt 的候选回答之间计算。分布式训练时，一个 prompt 的回答可能分散在不同设备上，需要先正确聚合全组奖励。若每张卡各自计算局部均值，改变 GPU 数量就可能改变训练目标。

### 9.2 组大小与采样多样性

较大的 $G$ 能提供更稳定的组统计量，也更可能同时采到成功和失败回答，但生成成本近似线性增加。若采样温度太低，同组回答高度重复，即使 $G$ 很大也未必增加有效信息；温度太高则可能产生大量无效文本。应同时记录组内唯一回答比例、奖励标准差和非零优势组比例。

### 9.3 标准差归一化不是无害操作

除以组标准差能让不同 prompt 的优势尺度更接近，但也会放大奖励差异很小的组，并改变不同难度问题的相对权重。若奖励是离散的，组内只有一个样本与其他样本不同，小标准差可能产生较大的标准化优势。应检查 `epsilon`、标准差定义、reward clipping 和全局梯度范数，而不是把归一化视为纯数值细节。

### 9.4 回答长度会改变样本权重

基础目标中的 $1/T_i$ 让每条回答获得近似相同总权重，但每个 token 分到的权重随回答长度变化。改为全局 token 平均后，长回答又会贡献更多 token。两种方式都可能形成长度偏好，因此应监控奖励与长度的相关性、正常结束率和截断比例。后续方法对长度归一化提出了多种修正，它们不是基础 GRPO 定义的一部分。

### 9.5 只优化有效回答 token

概率比、KL 和策略损失应使用 response mask 排除 prompt 与 padding。序列级优势需要广播到 token 维度，但不能广播到无效位置。对于达到最大长度而未正常结束的回答，要明确是保留任务奖励、增加超长惩罚，还是丢弃样本。

### 9.6 保持数据足够 on-policy

On-policy 的含义是：用于更新策略的数据，应由当前策略或与当前策略足够接近的策略生成。GRPO 的回答最初由 $\pi_{\mathrm{old}}$ 采样；在当前策略尚未明显改变时，这批数据仍近似 on-policy，可以配合概率比做有限次 minibatch 更新。

随着更新继续， $\pi_\theta$ 与 $\pi_{\mathrm{old}}$ 的差距增大，旧回答逐渐变成 stale data，不能再可靠代表当前策略会生成什么。概率比和 clipping 只能缓解小幅分布变化，不能支持无限复用旧样本。因此，GRPO 通常在一批 rollout 上更新有限次数，然后丢弃旧回答，用最新策略重新采样。应监控新旧策略 KL、clip fraction、概率比极值和每批更新后的奖励变化。

## 10. 常见问题与排查方向

| 现象 | 常见原因 | 优先检查 |
| --- | --- | --- |
| 大量组的优势全为 0 | 同组奖励完全相同；题目过易或过难 | 非零方差组比例、题目难度、采样温度 |
| loss 正常但策略几乎不变 | response mask 错误；有效组太少；KL 过强 | 有效 token、优势分布、梯度范数 |
| 奖励上涨但回答质量退化 | 奖励漏洞或格式奖励主导 | 各奖励分量、高奖励样本、独立评测 |
| 训练对 GPU 数量敏感 | 组统计只在局部设备计算 | prompt 分组索引、全局聚合逻辑 |
| 输出越来越长或越来越短 | 长度归一化与奖励共同造成偏差 | 长度—奖励相关性、token 聚合方式 |
| 大量 token 被裁剪 | 更新 epoch 或学习率过大；rollout 过旧 | clip fraction、新旧策略 KL |
| KL 突然增大 | 参考模型概率计算错误；KL 系数过小 | tokenizer、chat template、ref log probability |
| 组内回答几乎相同 | 温度过低；随机种子或生成参数错误 | 唯一回答比例、temperature、top-p |

尤其要检查策略模型、旧策略和参考模型是否使用完全一致的 tokenizer、chat template 与回答边界。模板不同会让 log probability 不可比，进而同时破坏概率比和 KL。

## 11. 如何评估 GRPO 模型

GRPO 评估不能只看训练 reward 或 loss，需要同时覆盖模型能力、奖励泛化、训练稳定性和组采样质量：

1. **训练稳定性**：观察 reward、参考模型 KL、clip fraction、梯度范数、回答长度和正常结束率是否稳定。
2. **绝对任务能力**：在隔离测试集上报告数学准确率、代码测试通过率、格式合规率等可解释指标。
3. **奖励泛化**：使用未参与训练的验证器、独立奖励模型或人工评价，检查是否只对训练奖励过拟合。
4. **能力回归**：检查通用问答、指令遵循、安全性、语言质量和原有领域能力是否退化。
5. **有效组比例**：记录奖励方差非零、能够提供相对优势的组占比。
6. **组内成功率分布**：统计每组全对、部分正确和全错分别占多少。
7. **组内多样性**：检查唯一回答数、推理路径差异和重复率。
8. **优势分布**：检查训练是否被少数低标准差组产生的极值主导。
9. **长度与奖励关系**：判断模型是否通过改变长度而非提高正确性获得收益。
10. **采样通过率**：在固定温度下比较 `pass@k`、单次准确率和多次采样成本。

训练时使用组内相对奖励，评测时仍应报告可解释的绝对指标。例如数学题准确率、代码测试通过率和人工偏好胜率，而不能只报告标准化 advantage 或训练 loss。

## 12. GRPO 的能力边界

GRPO 降低了 critic 的资源成本，但没有消除强化学习本身的困难：

- 它仍然需要在线或近在线生成，采样成本高；
- 它仍然依赖奖励质量，并可能出现 reward hacking；
- 组基线只比较同一 prompt 的有限样本，不是真实价值函数；
- 结果奖励下的 token 信用分配较粗糙；
- 当同组奖励缺少差异时，算法没有有效学习信号；
- 组归一化、长度聚合和 prompt 难度可能引入额外偏差。

因此，GRPO 不应被简单理解为“全面优于 PPO”。当任务能廉价地批量采样和可靠验证、critic 成本又很高时，GRPO 很有吸引力；当组采样昂贵、奖励噪声大，或需要更细粒度的状态价值估计时，PPO 或其他方法可能更合适。

## 13. 本章小结

- GRPO 对同一个 prompt 采样一组回答，用组内相对奖励代替 critic 估计优势。
- 它保留了 PPO 风格的新旧策略概率比、clipping、参考模型 KL 和 on-policy 更新。
- 结果监督下，同一回答的有效 token 通常共享序列级优势，因而信用分配比 GAE 更粗糙。
- 省掉 critic 能降低显存和训练复杂度，但需要多回答采样，并对组大小、奖励方差和采样多样性敏感。
- 零方差组、长度偏差、分布式分组错误和奖励漏洞是实践中必须重点监控的问题。

## 参考文献

1. Shao, Z., et al. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). 2024.
2. DeepSeek-AI, et al. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). 2025.
3. Schulman, J., et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017.
