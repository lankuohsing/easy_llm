# easy_llm

这是一个持续建设中的 LLM 学习仓库，计划覆盖预训练、中训练、后训练及相关实践。目前已有内容包括后训练方法的理论笔记与实验规划、tokenization 基础教程，以及预训练数据教程和面试问题清单；中训练的专门章节尚未开始整理。

训练阶段与常见实践路径见 [大语言模型训练全流程](./大语言模型训练全流程.md)。

## 当前内容

| 方向 | 入口 | 当前进度 |
| --- | --- | --- |
| 分词与 tokenizer | [大语言模型 Tokenization：方法、演进与实践](<./tokenization/大语言模型 Tokenization：方法、演进与实践.md>) | 主流方法、发展历程与工程核查 |
| 预训练 | [数据教程](<./pretrain/预训练数据：从来源到可训练语料.md>)、[面试问题清单](./pretrain/预训练面试问题清单.md) | 数据构建、Pile 实践与预训练岗位面试题 |
| 监督微调（SFT） | [sft/](./sft/readme.md) | 基础理论与实验规划 |
| 直接偏好优化（DPO） | [dpo/](./dpo/readme.md) | 基础理论 |
| 在线策略蒸馏（OPD） | [opd/](./opd/readme.md) | 目录占位，内容待补充 |
| 强化学习（RL） | [rl/](./rl/readme.md) | PPO、GRPO 基础理论 |
| 通用工具 | [utils/dataset_utils/](./utils/dataset_utils/) | 数据集下载与读取示例 |

## 待补充的目录

- [model_architecture/](./model_architecture/)：模型架构与关键组件，如注意力、位置编码和 MoE。
- [midtrain/](./midtrain/)：中训练与继续预训练，具体范围随专题材料确定。
- [evaluation/](./evaluation/)：模型能力、鲁棒性与安全评测。
- [inference/](./inference/)：模型推理与部署。

以上待补充目录目前仅有 `.gitkeep`，尚无教程正文。已有后训练方法的顶层目录和链接保持不变；开始撰写新专题时再添加名称明确的入口文档。理论笔记、实验规划和可运行示例会分别标明，避免把规划误认为已完成的实验。

文档格式约定见 [MARKDOWN_GUIDE.md](./MARKDOWN_GUIDE.md)。
