# easy_llm

这是一个持续建设中的 LLM 学习仓库，计划覆盖预训练、中训练、后训练及相关实践。目前已有内容包括后训练方法的理论笔记与实验规划、tokenization 基础教程，以及预训练数据教程和面试问题清单；中训练的专门章节尚未开始整理。

训练阶段与常见实践路径见 [大语言模型训练全流程](./大语言模型训练全流程.md)。

当前优先学习 [RL 基础 → PPO → GRPO](./rl/00_readme.md)。先掌握强化学习的核心概念和这两种策略优化方法，再决定 Agentic RL 的进阶学习与落地内容。

## 当前内容

| 方向 | 入口 | 当前进度 |
| --- | --- | --- |
| 分词与 tokenizer | [方法与实践](<./tokenization/01_大语言模型 Tokenization：方法、演进与实践.md>)、[BPE 演算](<./tokenization/02_bpe：从训练到编码与解码.md>)、[Pile 分词实践](<./tokenization/03_pile分词实践：从原始文本到训练与生成.md>) | 共同流程、设计取舍、BPE 演算，以及特殊 token、数据准备、训练与生成的衔接 |
| 预训练 | [数据教程](<./pretrain/02_预训练数据：从来源到可训练语料.md>)、[面试问题清单](./pretrain/04_预训练面试问题清单.md) | 数据构建、Pile 实践与预训练岗位面试题 |
| 监督微调（SFT） | [sft/](./sft/00_readme.md) | 基础理论与实验规划 |
| 直接偏好优化（DPO） | [dpo/](./dpo/00_readme.md) | 基础理论 |
| 在线策略蒸馏（OPD） | [opd/](./opd/00_readme.md) | 目录占位，内容待补充 |
| 强化学习（RL） | [rl/](./rl/00_readme.md) | RL 基础、PPO、GRPO 的顺序教程与数值核对；另有奖励机制和熵专题选读 |
| 通用工具 | [utils/dataset_utils/](./utils/dataset_utils/) | 数据集下载与读取示例 |

## 待补充的目录

- [model_architecture/](./model_architecture/)：模型架构与关键组件，如注意力、位置编码和 MoE。
- [midtrain/](./midtrain/)：中训练与继续预训练，具体范围随专题材料确定。
- [evaluation/](./evaluation/)：模型能力、鲁棒性与安全评测。
- [inference/](./inference/)：模型推理与部署。

以上待补充目录目前仅有 `.gitkeep`，尚无教程正文。已有后训练方法的顶层目录保持不变；开始撰写新专题时再添加 `00_readme.md` 入口文档。理论笔记、实验规划和可运行示例会分别标明，避免把规划误认为已完成的实验。

子目录中的文档按阅读顺序使用两位数编号：入口为 `00_readme.md`，正文从 `01_` 开始。按文件名字典顺序排序即可沿学习顺序阅读；没有入口文档的目录直接从 `01_` 开始。

文档格式约定见 [MARKDOWN_GUIDE.md](./MARKDOWN_GUIDE.md)。
