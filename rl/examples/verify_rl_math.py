"""用有限枚举、有限差分和独立表达式核对 RL / PPO / GRPO 教程。

运行：python3 rl/examples/verify_rl_math.py
只用标准库，不采样、不训练模型，也不创建输出文件。
"""

from itertools import product
from math import exp, isclose, log, sqrt


def close(actual, expected):
    if not isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
        raise AssertionError(f"实际值 {actual} 与预期 {expected} 不一致")


def derivative(function, x, step=1e-5):
    return (function(x + step) - function(x - step)) / (2 * step)


def sigmoid(theta):
    return 1 / (1 + exp(-theta))


def bernoulli(probability, outcome):
    return probability if outcome else 1 - probability


def paths(theta):
    """两步 MDP：环境转移随机，且转移概率不显式依赖 theta。"""
    probability = sigmoid(theta)
    for first_action, next_state, second_action in product((0, 1), repeat=3):
        transition = 0.2 + 0.6 * first_action
        path_probability = (
            bernoulli(probability, first_action)
            * bernoulli(transition, next_state)
            * bernoulli(probability, second_action)
        )
        rewards = (
            first_action,
            next_state + second_action + 2 * next_state * second_action,
        )
        scores = (first_action - probability, second_action - probability)
        yield path_probability, rewards, scores, next_state, (first_action, second_action)


def check_policy_gradient():
    theta, gamma = log(0.4 / 0.6), 0.9

    def objective(parameter):
        return sum(weight * (r[0] + gamma * r[1]) for weight, r, *_ in paths(parameter))

    full_gradient = reward_to_go_gradient = baseline_gradient = 0.0
    for weight, rewards, scores, state, _ in paths(theta):
        total_return = rewards[0] + gamma * rewards[1]
        full_gradient += weight * total_return * sum(scores)
        reward_to_go_gradient += weight * (
            total_return * scores[0] + gamma * rewards[1] * scores[1]
        )
        # 两个基线故意不准确；均在当前动作之前确定，并保持固定。
        baseline_gradient += weight * (
            (total_return - 7) * scores[0]
            + gamma * (rewards[1] - (-4 + 2 * state)) * scores[1]
        )
    exact_gradient = derivative(objective, theta)
    close(exact_gradient, 0.87936)
    for estimate in (full_gradient, reward_to_go_gradient, baseline_gradient):
        close(estimate, exact_gradient)
    print(f"[通过] 随机转移 MDP：真实目标有限差分、轨迹梯度、reward-to-go、状态基线 = {exact_gradient:.5f}")

    # 用旧状态和动作数据构造局部概率比目标；只在新旧参数相同时核对梯度。
    old_probability = sigmoid(theta)

    def local_surrogate(parameter):
        new_probability = sigmoid(parameter)
        total = 0.0
        for weight, rewards, _, _, actions in paths(theta):
            ratios = [
                bernoulli(new_probability, a) / bernoulli(old_probability, a)
                for a in actions
            ]
            total += weight * (
                ratios[0] * (rewards[0] + gamma * rewards[1])
                + gamma * ratios[1] * rewards[1]
            )
        return total

    close(derivative(local_surrogate, theta), exact_gradient)
    print("[通过] 旧数据上的局部概率比目标，在新旧策略相同时有正确的一阶梯度")
    if not (10 > 5 and 0.4 * 10 < 5):
        raise AssertionError("折扣应能反转两条路线的排序")
    print("[通过] 折扣反转：[5] 与 [0, 10] 在 gamma=1 / 0.4 下的优劣相反")


def gae(rewards, values, gamma, lam, terminated):
    """单段 rollout；尾部停止递推，只有真实终止才去掉尾部 bootstrap。"""
    advantages = [0.0] * len(rewards)
    residuals = [0.0] * len(rewards)
    tail = 0.0
    for t in reversed(range(len(rewards))):
        next_value = 0.0 if t == len(rewards) - 1 and terminated else values[t + 1]
        residuals[t] = rewards[t] + gamma * next_value - values[t]
        tail = residuals[t] + gamma * lam * tail
        advantages[t] = tail
    return advantages, residuals


def check_gae():
    rewards, values, gamma, lam = [0, 0, 1], [0.2, 0.4, 0.6, 0], 0.9, 0.8
    advantages, residuals = gae(rewards, values, gamma, lam, terminated=True)
    for actual, expected in zip(residuals, [0.16, 0.14, 0.4]):
        close(actual, expected)
    for actual, expected in zip(advantages, [0.46816, 0.428, 0.4]):
        close(actual, expected)

    # 独立用 k 步回报与有限 lambda 权重计算，而非再次调用递推式。
    for t in range(len(rewards)):
        remaining = len(rewards) - t
        mixture = 0.0
        weight_sum = 0.0
        for k in range(1, remaining + 1):
            k_step = sum(gamma**j * rewards[t + j] for j in range(k))
            k_step += gamma**k * values[t + k] - values[t]
            weight = (
                lam ** (k - 1)
                if k == remaining
                else (1 - lam) * lam ** (k - 1)
            )
            mixture += weight * k_step
            weight_sum += weight
        close(weight_sum, 1)
        close(mixture, advantages[t])
    zero_lambda, _ = gae(rewards, values, gamma, 0, terminated=True)
    one_lambda, _ = gae(rewards, values, gamma, 1, terminated=True)
    for t in range(len(rewards)):
        close(zero_lambda[t], residuals[t])
        mc_return = sum(
            gamma**j * rewards[t + j] for j in range(len(rewards) - t)
        )
        close(one_lambda[t], mc_return - values[t])
    # 网络即使对终止状态输出非零预测，也必须由终止标记清掉。
    terminal_nonzero, _ = gae(
        rewards, [0.2, 0.4, 0.6, 2], gamma, lam, terminated=True
    )
    for actual, expected in zip(terminal_nonzero, advantages):
        close(actual, expected)
    truncated, _ = gae(rewards, [0.2, 0.4, 0.6, 2], gamma, lam, terminated=False)
    for actual, expected in zip(truncated, [1.40128, 1.724, 2.2]):
        close(actual, expected)
    print("[通过] GAE 递推 = 有限 k 步混合；lambda=0/1；终止与截断 bootstrap")


def check_importance_sampling():
    target, behavior, reward = [0.5, 0.5], [0.01, 0.99], [1, 0]
    weighted = [p / q * r for p, q, r in zip(target, behavior, reward)]
    mean = sum(q * value for q, value in zip(behavior, weighted))
    variance = sum(q * (value - mean) ** 2 for q, value in zip(behavior, weighted))
    close(mean, sum(p * r for p, r in zip(target, reward)))
    close(variance, 24.75)
    print(f"[通过] 重要性采样保持期望 {mean:.2f}，但稀有样本权重 50 导致方差 {variance:.2f}")


def check_clipping():
    def objective(ratio, advantage):
        return min(ratio * advantage, min(1.2, max(0.8, ratio)) * advantage)

    # 越过有利方向边界时梯度为零；越过不利方向边界仍有纠正梯度。
    for ratio, advantage, value, slope in (
        (1.5, 2, 2.4, 0), (0.5, 2, 1, 2),
        (0.5, -2, -1.6, 0), (1.5, -2, -3, -2),
    ):
        close(objective(ratio, advantage), value)
        close(derivative(lambda x: objective(x, advantage), ratio), slope)
    print("[通过] PPO 裁剪的正负优势四种情况及有限差分梯度")


def check_group_baseline():
    p, group_size = 0.3, 3
    centered_gradient = leave_one_out_gradient = 0.0
    for outcomes in product((0, 1), repeat=group_size):
        probability = 1.0
        for outcome in outcomes:
            probability *= bernoulli(p, outcome)
        mean = sum(outcomes) / group_size
        centered_gradient += probability * sum(
            (r - mean) * (r - p) for r in outcomes
        ) / group_size
        leave_one_out_gradient += probability * sum(
            (r - (sum(outcomes) - r) / (group_size - 1)) * (r - p) for r in outcomes
        ) / group_size
    original_gradient = p * (1 - p)
    close(centered_gradient, (group_size - 1) / group_size * original_gradient)
    close(leave_one_out_gradient, original_gradient)
    rewards = [1, 0, 0, 0]
    mean = sum(rewards) / len(rewards)
    std = sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    normalized = [(r - mean) / std for r in rewards]
    expected_advantages = [sqrt(3), -1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)]
    for actual, expected in zip(normalized, expected_advantages):
        close(actual, expected)
    equal_rewards = [1, 1, 1, 1]
    equal_mean = sum(equal_rewards) / len(equal_rewards)
    close(sum((r - equal_mean) ** 2 for r in equal_rewards), 0)
    for r in equal_rewards:
        close((r - equal_mean) / 1e-8, 0)
    print("[通过] iid 组均值基线的 (G-1)/G 缩放、leave-one-out、总体标准差组优势")


def check_kl():
    theta, reference = log(0.4 / 0.6), [0.6, 0.4]

    def distribution(parameter):
        p = sigmoid(parameter)
        return [p, 1 - p]

    def k3(p, q):
        ratio = q / p
        return ratio - log(ratio) - 1

    def forward_kl(parameter):
        return sum(p * log(p / q) for p, q in zip(distribution(parameter), reference))

    def reverse_kl(parameter):
        return sum(q * log(q / p) for p, q in zip(distribution(parameter), reference))

    current = distribution(theta)
    close(sum(p * k3(p, q) for p, q in zip(current, reference)), forward_kl(theta))
    fixed_sample_gradient = sum(
        p * derivative(lambda x, i=i: k3(distribution(x)[i], reference[i]), theta)
        for i, p in enumerate(current)
    )
    close(fixed_sample_gradient, derivative(reverse_kl, theta))
    if isclose(fixed_sample_gradient, derivative(forward_kl, theta), abs_tol=1e-4):
        raise AssertionError("本例的固定样本 k3 反传不应等于正向 KL 梯度")
    print("[通过] k3 的标量期望 = 正向 KL；固定样本反传期望 = 反向 KL 梯度（固定状态）")


def main():
    checks = (
        check_policy_gradient,
        check_gae,
        check_importance_sampling,
        check_clipping,
        check_group_baseline,
        check_kl,
    )
    for check in checks:
        check()
    print("全部数值核对通过；这些有限例子不替代一般证明，也不代表实际模型训练结果。")


if __name__ == "__main__":
    main()
