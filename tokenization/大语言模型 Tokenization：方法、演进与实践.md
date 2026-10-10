# 大语言模型 Tokenization：方法、演进与实践

## 摘要

Tokenization 将文本转换为模型使用的 token ID 序列。切分规则同时影响词表覆盖、序列长度、训练效率、推理成本及多语言使用体验。本文区分子词算法与分词框架，概述 BPE、WordPiece、Unigram 和字节级 BPE 的原理，梳理 GPT-3 以来代表模型的选择，并给出评估和使用 tokenizer 的基本方法。模型词表的大小与实现细节应以对应版本的官方模型文件为准；不同研究的压缩率数据不能脱离语料和统计口径直接比较。

## 1. 范围与基本概念

本文以文本生成模型为主，也介绍对预训练分词方法有重要影响的 WordPiece。典型处理链包括文本归一化、预分词、子词切分、特殊 token 处理以及 token 到 ID 的映射。并非每个实现都显式包含全部环节。模型输入的是 ID 序列；词表和嵌入矩阵决定这些 ID 的含义。因此，训练与推理应使用与模型权重配套的 tokenizer 文件；对聊天模型还须使用配套的聊天模板。

词级切分容易遇到未登录词；纯字符或纯字节切分覆盖面广，但通常产生较长序列。子词方法用有限词表表示常见片段，并把罕见词拆成更小的单位。以下须区分两个层次：**BPE、WordPiece、Unigram 是词表学习与切分方法；SentencePiece 是可运行 BPE 或 Unigram 的框架；字节级 BPE 是以字节为基础符号的 BPE 变体。**

## 2. 主流方法

### 2.1 BPE

BPE 最初用于数据压缩，Sennrich 等将它用于神经机器翻译的开放词表问题 [1]。训练时，先把语料拆成基础符号，统计相邻符号对的频次，反复合并最高频的一对，直到达到合并预算。训练结果包含有序合并规则；编码新文本时按规则优先级合并，**不再根据新文本重新统计频次**。原始子词 BPE 通常从字符出发，未覆盖的字符仍可能需要未知符号或回退机制。

例如，语料若频繁出现 `low`，算法可能依次学到 `l` 与 `o`、再把 `lo` 与 `w` 合并；`lower` 即使未成为单独 token，仍可由已学子词表示。这个例子只说明机制，实际切分由训练语料、预分词规则和合并次序共同决定。

### 2.2 WordPiece

WordPiece 常见于 BERT 系模型。编码阶段通常对预分词后的每个词执行从左到右的最长匹配；BERT 约定用 `##` 标记词内续接片段。如果某个位置无法匹配，BERT 风格实现可能将整个词映射为 `[UNK]` [2]。

公开教程常用下式解释 WordPiece 与“合并最高频相邻对”的 BPE 有何不同：

$$
\mathrm{score}(a,b)=\frac{\mathrm{count}(a,b)}{\mathrm{count}(a)\mathrm{count}(b)}
$$

该式倾向于合并相对稳定地共同出现的片段。不过，Google 未公开原始 WordPiece 训练实现；Hugging Face 明确将上述过程称为依据论文所作的教学复现 [2]。因此，该式应视为常用复现准则，而非已验证的原始算法或精确的似然增益等式。

### 2.3 Unigram

Unigram 从较大的候选词表出发，为每个子词估计概率。若一段文本可以有多种合法切分，其概率是各切分概率之和：

$$
P(x)=\sum_{s\in\mathcal{S}(x)}\prod_{t\in s}p(t)
$$

这里， $\mathcal{S}(x)$ 表示文本的合法切分集合。训练通过迭代估计子词概率，优先删除**移除后使语料似然下降较小**的候选，并保留损失较大的候选 [3][4]。常规编码以 Viterbi 算法取得概率最高的切分；训练语言模型时也可对多种切分采样，进行子词正则化 [3]。

### 2.4 字节级 BPE 与 SentencePiece

GPT-2 谱系的字节级 BPE 以 256 个字节值为基础符号，再学习 BPE 合并；只要输入可编码为字节，就不会因遇到词表外字符而必须使用未知 token。UTF-8 汉字通常占多个字节，但常用字节串可以被合并为一个 token。**字节数不等于最终 token 数**，后者仍取决于语料、预分词正则和合并表 [5][6]。

SentencePiece 直接从原始文本训练，可选 BPE 或 Unigram 后端；它用 `▁` 表示空格，避免把按空格切词设为前提 [4]。空格实体化有助于还原切分后的文本，但默认的 `nmt_nfkc` 归一化、多余空格折叠，以及未覆盖字符的处理，都可能改变原始输入。因此，不能笼统保证解码结果与任意原文逐字节相同 [4]。

| 方法 | 主要训练思路 | 常见编码方式 | 需要注意的边界 |
| --- | --- | --- | --- |
| BPE | 逐次合并高频相邻对 | 按有序合并规则切分 | 字符覆盖和预分词取决于实现 |
| WordPiece | 公开复现常用相对共现打分 | 最长匹配 | 原始训练细节未完全公开；可能产生 `[UNK]` |
| Unigram | 估计子词概率并逐步剪枝 | 最优切分或采样 | 采样主要用于训练期，推理通常采用确定性切分 |
| 字节级 BPE | 从字节出发学习 BPE 合并 | 预分词后按规则合并 | 无未知字节不等于各语言压缩率相同 |

## 3. 从 GPT-3 开始的发展

GPT-3 沿用 GPT-2 的字节级 BPE。此后的变化既包括词表规模，也包括预分词规则和多语言语料分配；并不存在所有模型单调扩表的共同路径。下表列出代表性实例，数字按各来源口径给出，**不应把 tokenizer 的实际词条数与模型配置中的嵌入行数混为一谈**。

| 时间与模型 | 主要选择 | 公开词表规模 | 依据 |
| --- | --- | --- | --- |
| 2020，GPT-3 | 沿用 GPT-2 字节级 BPE | 50,257 | [5][6] |
| 2022，BLOOM | 面向多语言训练的字节级 BPE | 约 250K | [7] |
| 2023，LLaMA 1/2 | SentencePiece BPE | 32K | [8] |
| 2023，GPT-3.5/GPT-4 | `cl100k_base`，调整词表与预分词规则 | 100,277 | [6] |
| 2024，GPT-4o | `o200k_base`，改善多语言压缩 | 200,019 | [6][9] |
| 2024，Llama 3 | 转向 tiktoken 风格的字节级 BPE | 约 128K | [10] |
| 2024，Mistral NeMo | 从早期 SentencePiece 系转向 Tekken | 约 131K 常规 token | [11] |
| 2025，Qwen3 | 字节级 BPE | 151,669 | [12] |

上表说明：不能仅凭品牌判断 tokenizer，例如 LLaMA 1/2 与 Llama 3、早期 Mistral 与 Tekken 所用方案不同；2022 年 BLOOM 的大词表也说明“词表只会随年份增长”不成立。词表大小需要权衡文本压缩、嵌入层参数量、长尾 token 的训练频次以及目标语言分布。不存在脱离模型规模、训练数据和任务的通用最优词表区间。

## 4. 多语言与新方向

同一语义内容在不同语言中的 token 数可能明显不同，影响按 token 计费、上下文可容纳的文本量及推理工作量 [13]。原因包括 UTF-8 编码长度、语料采样比例、预分词边界和合并表覆盖。字节级方法保证可表示性，却**不保证**各语言有相同的 token 压缩率。比较 tokenizer 时，应在同一批平行语料上分别编码，说明是否计入特殊 token，并报告每字符或每字节的 token 数、文本类型及统计量；不同文章测得的“中文相对英文倍数”不可直接拼成时间趋势。压缩率也不能单独证明模型质量更好；下游效果、训练充分性和实际吞吐须另行评估。

Byte Latent Transformer（BLT）通过动态分块减少全局模型处理的字节序列长度 [14]；H-Net 进一步研究可学习的分块边界 [15]。这些工作为摆脱固定子词边界提供了路径，但字节级基线的长度和计算成本受具体语料及模型结构影响，不能一概写成“序列必增四倍、总计算必增十六倍”。从已公开的主流生成式大模型看，固定子词 tokenizer 仍是常见选择；新方法是否在前沿规模和部署成本上持续占优，仍需同口径验证。

## 5. 工程核查与最小示例

1. **以目标模型为准**：同时保存 tokenizer、特殊 token 定义、归一化规则和聊天模板，并固定模型与 tokenizer 的版本。tiktoken 可支持自定义编码，但其内置编码不能直接用于不匹配的模型 [6]。
2. **在 ID 层检查数据**：抽查中文、空白、数字、emoji、代码和特殊 token 字符串的编码结果。需要检查原文往返时，先确认 tokenizer 是否进行了不可逆归一化；聊天角色和边界应由模板生成，而非由用户文本直接拼接。
3. **在编码后控制长度**：按实际 token 数截断或分块。纯文本计数不等于完整聊天请求的计数，消息边界、工具描述和多模态输入可能产生额外开销。
4. **区分扩词表与换 tokenizer**：预训练后仍可增加普通或特殊 token，并调整嵌入矩阵、继续训练；整体更换切分规则会改变原有 ID 的含义，须重新对齐并充分验证，成本通常更高 [16]。
5. **完整保存自训产物**：单个 `tokenizer.json` 不必然能由 `AutoTokenizer.from_pretrained` 识别。应保存相应的 tokenizer 配置，或显式用 `PreTrainedTokenizerFast(tokenizer_file=...)` 读取该文件 [16]。

以下示例比较**同一段文本**在 OpenAI 三代公开编码下的切分数量，不把结果外推为跨语言质量结论。运行需要 `tiktoken`；首次使用编码时可能需要联网下载词表。

```python
import tiktoken

text = "人工智能正在改变软件开发。"
for name in ("r50k_base", "cl100k_base", "o200k_base"):
    encoding = tiktoken.get_encoding(name)
    token_ids = encoding.encode(text)
    assert encoding.decode(token_ids) == text
    print(name, len(token_ids), token_ids)
```

## 6. 结论

Tokenization 的关键设计不止“选 BPE 还是 Unigram”，还包括文本归一化、预分词、词表训练语料、特殊 token 与模型版本。BPE 系方法因压缩率和成熟工具链而广泛使用；Unigram 提供概率切分与训练期采样；SentencePiece 提供语言无关的训练框架。评估一个 tokenizer 应同时检查覆盖、压缩率、目标任务表现和工程可复现性，并用同一语料及统一口径比较候选方案。

## 参考文献

1. Sennrich, Haddow, Birch. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/). ACL, 2016.
2. Hugging Face. [WordPiece tokenization](https://huggingface.co/learn/llm-course/en/chapter6/6). LLM Course.
3. Kudo. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://aclanthology.org/P18-1007/). ACL, 2018.
4. Kudo, Richardson. [SentencePiece](https://aclanthology.org/D18-2012/). EMNLP, 2018；另见 [官方归一化说明](https://github.com/google/sentencepiece/blob/master/doc/normalization.md) 与 [Unigram 训练源码](https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc)。
5. Brown et al. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS, 2020.
6. OpenAI. [tiktoken 编码定义](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)、[模型映射](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py)及[自定义编码说明](https://github.com/openai/tiktoken#extending-tiktoken).
7. BigScience Workshop. [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100). 2022.
8. Touvron et al. [LLaMA](https://arxiv.org/abs/2302.13971) 与 [Llama 2](https://arxiv.org/abs/2307.09288). 2023.
9. OpenAI. [Hello GPT-4o](https://openai.com/index/hello-gpt-4o/). 2024.
10. Llama Team. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). 2024.
11. Mistral AI. [Tokenization and chat templates](https://docs.mistral.ai/resources/cookbooks/concept-deep-dive-tokenization-chat_templates). 官方文档.
12. Qwen Team. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). 2025.
13. Petrov et al. [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425). NeurIPS, 2023.
14. Pagnoni et al. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871). 2024.
15. [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955). arXiv:2507.07955, 2025.
16. Hugging Face. [添加普通 token](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_tokens)、[调整模型嵌入矩阵](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings)及 [AutoTokenizer 实现](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/tokenization_auto.py). Transformers 文档与源码.
