# Pile 分词实践：从原始文本到训练与生成

本文把分词学习中的基本概念与实际数据流程连接起来：原始 JSONL 如何变成 token ID，词表与合并规则如何配合，正文里的特殊 token 字符串怎样处理，以及这些选择在训练和生成时分别产生什么作用。

实践实例采用 [train-llm-from-scratch](https://github.com/lankuohsing/train-llm-from-scratch) 中的 Pile 数据流程。脚本行为依据 2026-10-30 核查的版本 `b6b439f`；文中的命令在该训练仓库根目录执行，路径使用可替换的占位值。本文所在的 `easy_llm` 仓库用于保存教材，不包含这些训练脚本。

设计思想与不同方法的对比见 [Tokenization：方法、演进与实践](<./01_大语言模型 Tokenization：方法、演进与实践.md>)，BPE 的逐轮演算和教学实现见 [BPE：从训练到编码与解码](<./02_bpe：从训练到编码与解码.md>)，语料来源与下载见 [Pile-uncopyrighted](../pretrain/01_pile-uncopyrighted.md)。

## 1. 先把原始文本、token 和模型输入分清楚

原始 Pile 文件是 `.jsonl` 或 zstd 压缩的 `.jsonl.zst`。解压后每行是一条 JSON 记录，主要字段如下：

```json
{"text": "文档正文……", "meta": {"pile_set_name": "ArXiv"}}
```

`text` 是尚未分词的字符串，`meta` 是来源等元数据。JSONL 是记录格式，zstd 是压缩格式，它们都不决定 tokenizer 的切分方式。一个 JSONL 记录也不等于一个固定长度的训练样本。

当前实践流程为：

```text
原始 .jsonl / .jsonl.zst
→ 读取或流式解压
→ 解析 JSON，取 text
→ r50k_base 编码正文
→ 在每条文档后追加 EOT
→ 拼成一维 token 流，写入 HDF5
→ 训练时取固定长度窗口、右移标签
→ Transformer 预测下一 token
```

| 对象 | 它是什么 | 与其他概念的区别 |
| --- | --- | --- |
| 原始文档 | JSONL 的一条 `text` 字符串 | 不等于一个 token，也不必恰好装入一个训练窗口 |
| token | 分词器词表中的一个单位 | 不一定是完整词，也不一定是完整 Unicode 字符 |
| token ID | 该单位的整数编号 | 不表示语义距离，也不是 Unicode 码点 |
| HDF5 的 `tokens` | 依次拼接的整数数组 | 不再保存原始 JSON 结构、来源标签或独立文档 offset |
| 训练样本 | 从整数流中取得的一段窗口 | 可以跨文档，也可以只包含长文档的一部分 |

## 2. BPE、`r50k_base` 与 GPT-2 的对应关系

### 2.1 训练规则和使用规则是两个过程

BPE 训练从基础符号开始，统计当前相邻符号对的频次，合并最高频 pair，记录顺序，再重新统计。常规编码时，已经冻结词表和优先级，不根据新输入重新计算频次或临时添加词条。

```text
训练：语料 → 统计与逐轮合并 → 词表 + 合并优先级
编码：新文本 → 按固定规则合并 → token → ID
解码：ID → token 对应内容 → 拼接与还原
```

BPE 编码不等于词表最长匹配；解码通常也不需要倒放合并步骤，因为反向词表已经保存了最终 token 的完整内容。

GPT-2 的字节级 BPE 从 256 个基础字节 token 出发，再产生 50,000 个合并 token，另有一个 `<|endoftext|>`，合计 50,257 个 ID。普通 token 有 50,256 个，特殊 token 的 ID 为 `50256`。

### 2.2 词表项 `"ĠField": 7663` 如何读

这是“内部 token 字符串 → ID”的映射。GPT-2 把字节可逆映射为内部 Unicode 字符，`Ġ` 代表空格字节 `0x20`，`Ċ` 代表换行字节。因此：

```text
原文 " Field"   → [7663]
原文 "Field"    → [15878]
原文 "ĠField"   → [128, 254, 15878]
```

第一项带一个前导空格。第三项真正包含 Unicode 字符 `Ġ`，编码的是它自己的 UTF-8 字节，不会把它当作空格。

在配套的 Hugging Face GPT-2 tokenizer 中，`convert_tokens_to_ids("ĠField")` 是直接查询内部词表项；`encode(" Field")` 是从原始文本执行分词。两种调用处理的对象不同。

```python
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained(
    "/path/to/gpt2", local_files_only=True, add_prefix_space=False
)
assert tokenizer.convert_tokens_to_ids("ĠField") == 7663
assert tokenizer.encode(" Field", add_special_tokens=False) == [7663]
assert tokenizer.encode("Field", add_special_tokens=False) == [15878]
assert tokenizer.decode([7663], clean_up_tokenization_spaces=False) == " Field"
```

### 2.3 合并优先级和词表编号各自做什么

Hugging Face GPT-2 的 `tokenizer.json` 中，`model.merges` 保存有序合并规则，`model.vocab` 保存词表编号。例如：

| 合并规则 | 规则下标，从 0 开始 | 合并结果与 ID |
| --- | --- | --- |
| `Ġ t` | 0 | `Ġt` → 256 |
| `h e` | 2 | `he` → 258 |
| `Ġt he` | 6 | `Ġthe` → 262 |

数组下标越小，优先级越高。文本 `" the"` 在一个预分词片段内依次变为：

```text
Ġ t h e → Ġt h e → Ġt he → Ġthe → ID 262
```

原版 GPT-2 的合并结果 ID 恰好等于规则下标加 256，但这不是所有 BPE 都保证的编号规律。tiktoken 的 `r50k_base` 通过“字节串 → rank/ID”表达规则，例如 `b" the" → 262`；其 rank 同时是普通 token ID，与 GPT-2 合并表下标 `6` 的数值不同，合并优先顺序一致。

### 2.4 两个库相同到什么程度

原版 GPT-2 与 `r50k_base` 的普通词表、ID 和合并优先级对应相同。比较文件时，应还原 GPT-2 的字节映射后再比较，不能直接拿内部显示字符与 tiktoken 的 `bytes` 比。

在已核查的本地文件中，全部 50,256 个普通词条和 50,000 条合并规则对应一致；此前的普通文本、小型 Pile 样本和随机文本检查也未发现 ID 差异。这支持普通编码的对应关系，**不能代替对特殊 token、前导空格、额外 BOS/EOS、截断和解码选项的检查**。

模型配置中的词表维度也不一定等于 tokenizer 的词条数。该训练项目使用 `vocab_size=50304`，大于实际 50,257 个编码 ID；多出的输出类别不是新增的 tokenizer 词条，不能直接交给 `r50k_base` 解码。检查数据 ID 合法性，应以 tokenizer 的定义为准。

## 3. EOT、EOS 和正文中的标记字符串

### 3.1 名称、角色与 ID 分开理解

| 名称 | 通常含义 | 当前 GPT-2 实例 |
| --- | --- | --- |
| EOT：End of Text | 文本结束 | tiktoken 的 `eot_token` 为 `50256` |
| EOS：End of Sequence | 模型接口中的序列结束角色 | HF GPT-2 的 `eos_token_id` 为同一个 `50256` |
| EOT：End of Turn | 某些聊天模型中的消息或回合结束 | 与当前 tiktoken 的 EOT 命名不同，不能直接类推 |

当前项目中，EOT 与 EOS 指向同一个 `<|endoftext|>`。但其他模型可以使用不同结束标记。例如原版 Llama 3 Instruct 的 `<|eot_id|>` 表示消息/回合结束，与 `<|end_of_text|>` 是不同 token；结束一条消息也不表示后续不能继续多轮对话。

特殊 token 的角色由模型配置和训练协议定义，不由名字或最大 ID 推断。HF GPT-2 还可能把同一个 token 配置为 BOS 或 UNK；存在 UNK 角色不代表普通未知字符会被丢弃，因为其基础字节词表覆盖有效 UTF-8 输入。

### 3.2 原始 JSONL 中只有可见字符串

例如：

```json
{"text": "A<|endoftext|>B"}
```

正文中的 `<|endoftext|>` 只是字符串。它是否变成特殊 ID，取决于 tokenizer 配置和调用方式；它也可能是教程、代码或引用内容。

`r50k_base` 不会自动把其他模型的 `</s>`、`[EOS]` 等字符串变成 `50256`。当前预处理脚本读取 `text` 字符串，也不把记录中另外存放的 `input_ids` 当作已经分词的输入。

### 3.3 当前脚本的编码策略

`data_preprocess.py` 的核心逻辑是：

```python
ids = encoder.encode_ordinary(text)
ids.append(encoder.eot_token)
```

`prepare_pretrain_data.py` 调用批量版本 `encode_ordinary_batch`，语义相同。正文里的标记按普通文字编码，真实文档结尾再追加一个 EOT：

```text
原始正文：A<|endoftext|>B
编码结果：A → 表示标记字符串的多个普通 token → B → 50256
```

如果原文末尾已经写着这串字符，当前逻辑仍把它作为正文，并追加一个真正的 EOT；不会单凭这串字符就跳过末尾追加。

对 Pile 原始记录，这种策略让文档边界来自 JSONL 记录结构，正文引用保留为正文。若来源格式明确规定该字符串是结构分隔符，应先按来源协议解析，再组装正文与边界 ID，不能对所有来源使用同一种猜测。

### 3.4 其他调用怎样处理正文标记

下面限于 GPT-2 / `r50k_base`，其他 HF tokenizer 应检查自己的特殊 token 定义：

| 调用方式 | 正文中 `<\|endoftext\|>` 的处理 |
| --- | --- |
| tiktoken `encode_ordinary(text)` | 作为普通文字，得到多个普通 ID |
| tiktoken `encode(text)` 默认调用 | 遇到已知特殊 token 字面量会报 `ValueError` |
| tiktoken `encode(text, disallowed_special=())` | 禁止检查关闭，未显式允许的标记按普通文字编码 |
| tiktoken `encode(text, allowed_special={"<\|endoftext\|>"})` | 显式允许，编码为 `50256` |
| HF `encode(text, add_special_tokens=False)` | 默认仍识别正文标记为 `50256` |
| HF `encode(text, add_special_tokens=False, split_special_tokens=True)` | 按普通文字切分，可与当前 ordinary 策略对齐 |

两个 HF 参数的职责不同：`add_special_tokens=False` 控制是否额外添加模型边界 token，`split_special_tokens=True` 在本例中控制正文特殊 token 字面量是否按普通文字切分。GPT-2 默认编码通常不自动追加末尾 EOS；其他模型的 BOS/EOS 包装行为需要分别核查。

### 3.5 替换为 Hugging Face 时的等价写法

下面代码需要本地原版 GPT-2 tokenizer 文件、`transformers` 与 `tiktoken`。tiktoken 首次加载可能需要联网获取并缓存词表。

```python
import tiktoken
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained(
    "/path/to/gpt2", local_files_only=True, add_prefix_space=False
)
encoding = tiktoken.get_encoding("r50k_base")
text = "Hello <|endoftext|> world"

hf_ids = tokenizer.encode(
    text, add_special_tokens=False, split_special_tokens=True
)
ordinary_ids = encoding.encode_ordinary(text)
assert hf_ids == ordinary_ids

assert tokenizer.eos_token_id == encoding.eot_token == 50256
hf_ids.append(tokenizer.eos_token_id)
ordinary_ids.append(encoding.eot_token)
assert hf_ids == ordinary_ids
```

此接口组合在 `transformers 5.3.0`、`tokenizers 0.22.2`、`tiktoken 0.12.0` 下核查。显式加载 `GPT2TokenizerFast` 可确定模型角色；仅用 `AutoTokenizer` 加载不完整配置时，可能得到通用后端且没有设置 `eos_token_id`。更换版本或配置后，应重新检查 ID 和角色。

不要将 EOT 字符串拼到正文后，再使用 `encode_ordinary`，因为这样得到的仍是普通字符序列。需要真正的结构 token 时，应追加对应 ID。

## 4. 实际训练时，特殊 token 会产生什么效果

### 4.1 训练样本是右移的 token 窗口

当前 loader 从连续 HDF5 token 流中划定固定长度窗口，再打乱窗口读取顺序。取得长度为 `T+1` 的片段后：

```text
原片段：[a1, a2, EOT, b1, b2]
输入 x：[a1, a2, EOT, b1]
目标 y：[a2, EOT, b1, b2]
```

这里 `a1、a2、b1、b2` 是示意 token，EOT 对应 `50256`。训练既监督模型在文档 A 末尾预测 EOT，也监督它在 EOT 之后预测文档 B 的首 token。

### 4.2 EOT 不会自动停止 loss、隔离注意力或重置位置

当前预训练模型对全部目标位置计算交叉熵，没有忽略 EOT，也不在 EOT 后关闭 loss。注意力只使用因果下三角掩码，阻止看未来；同一窗口内，B 仍可以注意到 A。

位置编号从每个模型输入窗口的起点计数，不会因为出现 EOT 重置。模型可以从训练中学习“这里通常开始新的文档”，但代码没有强制把它之前的信息清空。

| 想实现的行为 | 除了结束 ID，还需要什么 |
| --- | --- |
| 让模型学习文档结尾 | 在真实结尾提供结束 ID，并把该位置纳入预测目标 |
| 禁止跨文档注意力 | 按文档边界构造相应 attention mask 或独立样本 |
| 不预测下一文档开头的跨边界目标 | 单独设计标签或 loss mask |
| 重置文档内位置 | 明确提供或计算相应位置编号 |
| 输出结束 ID 后停止生成 | 在生成循环里检查结束 ID |

这些配置可以组合，但不是 EOS/EOT 自动附带的动作。SFT 中还可能只监督 assistant 回复等位置；当前项目的聊天数据使用相应 loss mask，不能把预训练的全位置 loss 结论直接套到全部后训练流程。

### 4.3 正文策略怎样改变监督信号

| 策略 | 示例数据 | 模型在该处学习什么 |
| --- | --- | --- |
| ordinary 正文 + 文档末尾 EOT | `A → 标记的普通字符序列 → B → EOT` | 写出正文里的可见标记，并在真实文档结尾预测 EOT |
| 识别正文标记为特殊 ID + 末尾 EOT | `A → EOT → B → EOT` | 在正文标记位置预测 EOT，同时继续预测其后的 B |

如果标记只是引用，第二种策略增加了与真实文档结尾不一致的 EOT 监督；如果标记本来就是有效分隔符，第一种策略又会把结构当作正文。选择应由数据语义决定。

若正文末尾的 EOT 已被识别为特殊 ID，再无条件追加，就可能得到 `EOT → EOT`，并训练模型预测重复结束符。重复边界、内文边界和 ordinary 编码还会改变 token 数，因此使用固定 token 预算时，也可能改变实际处理的原文量。

影响大小取决于出现频率、上下文、训练目标与推理策略。本文确认了输入和监督信号怎样改变，没有进行质量对照训练，不能给出“哪种一定更好”或提前停止概率的结论。

## 5. 生成停止与解码是另外两层逻辑

### 5.1 当前项目的两个生成入口

| 入口 | 对新生成的 `50256` 如何处理 |
| --- | --- |
| `scripts/generate_text.py` → `Transformer.generate` | 固定循环 `max_new_tokens`，不因 EOT 停止 |
| `scripts/chat.py` / 公共评测生成函数 | 配置 `stop_tokens=(50256,)`，新生成 EOT 后停止该样本，并从展示文本中移除终止符 |

同一个已训练模型，生成循环不同，表现也可以不同。提示词里已有 EOT 不会直接触发当前聊天生成器停机；它检查的是新生成的 token。

如果停止条件按 ID 检查，模型生成普通字符序列 `<|endoftext|>` 不会触发 `50256` 停止；生成一个真正的 EOT 才会命中该条件。只有另外实现了字符串停止规则时，可见字符串才可能单独触发停止。

### 5.2 解码结果相同，不足以证明编码相同

下列两种 ID 序列都可能解码为同一串可见文字：

```text
[27, 91, 437, 1659, 5239, 91, 29] → "<|endoftext|>"
[50256]                         → "<|endoftext|>"
```

前者是普通字符序列，后者是特殊 token。在 Hugging Face 的 `decode(..., skip_special_tokens=True)` 中，真正的特殊 ID 会被跳过，普通字符拼成的同名文字会保留。tiktoken 的 `decode` 没有这个参数；当前项目的聊天解码包装层通过显式筛选 ID 移除结束标记和没有词表项的多余输出类别。因此应同时检查 token ID、特殊 token 配置和解码选项。

同样，`decode(encode(text)) == text` 不保证任意 ID 序列满足 `encode(decode(ids)) == ids`：多种切分可以表示同一文本，重新编码会选择当前规则规定的路径。

字节级解码应先拼接各 token 的字节，再进行 UTF-8 解码。中文和 emoji 的单个 token 可能只包含一个字符的部分字节；逐 token 解码可能产生错误或 `�` 替换字符。对于任意截断的 ID 片段，也不能保证最后恰好是完整字符。

## 6. 两个数据准备脚本是替代方案

`data_preprocess.py` 与 `prepare_pretrain_data.py` 都得到同一种 HDF5 token 流。已经运行前者完成分词，不需要再把其 HDF5 交给后者；后者也不读取这个 HDF5 作为输入。

| 项目 | `data_preprocess.py` | `prepare_pretrain_data.py` |
| --- | --- | --- |
| 职责 | 处理已经下载的原始文本 | 下载指定 Pile 文件，再处理 |
| 原始输入 | 指定目录中的 `.jsonl` / `.jsonl.zst`，文件名排序 | 指定 `.jsonl.zst`，已有预期缓存文件时复用 |
| 默认范围 | 训练与验证；可用 `--val_only` | 每次通过 `--split train/val` 指定一种 |
| 默认处理量 | 每个文件前 1,000 行；`--all_data` 才全量 | 所选文件默认全量；可传 `--max_tokens` |
| 编码 | 每条 ordinary 编码 | 每 1,024 条 ordinary 批量编码 |
| 写缓冲 | 默认约 100 万 token，可配置 | 约 800 万 token，代码固定 |
| token 存储 | 一维 `tokens`，int32 | 一维 `tokens`，int32 |
| HDF5 输出保护 | 写 `.part`，成功后原子替换正式路径 | 直接写正式文件，中断可能留半成品 |
| 分词断点恢复 | 没有；重跑重新处理 | 没有；重跑重新处理 |
| 调试 JSON | 可以额外导出完整验证 token 序列 | 无对应选项 |

这里的“前 1,000 行”包括被解析或文本检查跳过的行，不保证最终写入 1,000 条有效文档。两个脚本都采用缓冲写入。批量分词有 CPU 并行机会，但需要更多文档和 token 缓冲内存，实际速度应测量。

`prepare_pretrain_data.py` 还有两个与正式处理相关的限制：

- `--max_tokens` 检查的是已经写出的 token，没有计入尚未写出的缓冲，不是严格上限；设置 100 万时可能到首次写出约 800 万甚至更多才停止。它先下载全部请求分片，因此这个参数也不限制下载量。
- 下载地址默认是 Hugging Face 官网；没有超时、自动重试或 Range 续传，失败后重跑从零下载；正式缓存只检查文件是否存在。它没有复用独立下载脚本中的镜像与续传机制。在中国大陆，应先核查实际下载入口和最终重定向域名。

对已经下载原始文件的学习与调试场景，使用 `data_preprocess.py` 更直接。两者在相同有效文档、相同顺序、相同 tokenizer 和处理范围下，token 内容可以一致；HDF5 文件本身的二进制字节不必相同，因为 chunk 布局等存储设置不同。

## 7. 读取、存储与正式处理时的要点

### 7.1 流式解压不产生独立的解压文件

`data_preprocess.py` 用 `zstd.open(..., "rt")` 读取 `.jsonl.zst`，随读取逐步解压，最终落盘的是 token HDF5，不另存一个解压后的 JSONL。未压缩的 `.jsonl` 则直接打开。

流式读写不是 zstd 独有，gzip、bz2、lzma 也提供类似接口。流式处理减少了整份语料常驻内存和解压副本的空间，但仍要容纳当前整条 JSONL 文档、分词结果及写缓冲；遇到超长文档，内存占用可以明显增加。

HDF5 的 chunk 大小是存储和写入单位，不等于 Transformer 的 `context_length`，也不是文档长度。缓冲达到阈值后才写入，可以减少扩容和写入调用；一条超长文档也可能让缓冲超过该阈值。

### 7.2 分词产物与原文分开保存

`tokens` 使用 int32，主要数据体积约为 `4 × token 数` 字节，另有 HDF5 元数据和 chunk 分配开销。不能用压缩原始文件的大小直接推算 token 数；需要通过小样本测量和实际累计统计来规划空间。

当前输出不记录 tokenizer 名称、版本和来源标签。实践中应另外保留这些信息、输入清单和处理范围，否则单凭一个 `.h5` 很难确认它是完整数据还是抽样数据，以及该用哪个 tokenizer 解码。

同一目录如果同时保留同一分片的 `.jsonl` 和 `.jsonl.zst`，当前文件扫描会把两者都处理，造成重复。输入目录应保存清楚的文件集合；训练、验证和测试的划分也应保留，两个预处理脚本当前没有独立测试集 CLI 入口。

`--debug_tokens_json` 会把完整 token 数组读入内存，再写 JSON，只适合小样本。正式语料仍以 HDF5 为主，检查时读小切片，不要为了查看前几个 token 加载全部数组。

### 7.3 小样本和全量命令

以下命令在训练仓库根目录执行。分词需要 Python、`h5py`、`numpy`、`tiktoken`、`zstandard`、`tqdm`，主要使用 CPU 和内存，不要求 GPU。解释器与依赖应属于同一个环境。

处理本地验证小样本，并额外输出可查看的 JSON：

```bash
python scripts/data_preprocess.py \
  --val_only \
  --val_dir /path/to/pile/val_small \
  --out_val_file /path/to/tokens/pile_val_small.h5 \
  --debug_tokens_json /path/to/tokens/pile_val_small_tokens.json \
  --all_data
```

处理已下载的完整训练与验证文件，不导出完整 JSON：

```bash
python scripts/data_preprocess.py \
  --train_dir /path/to/pile/train \
  --val_dir /path/to/pile/val \
  --out_train_file /path/to/tokens/pile_train.h5 \
  --out_val_file /path/to/tokens/pile_dev.h5 \
  --all_data
```

`--all_data` 表示处理输入目录实际包含的文件，不会自动下载缺失的训练分片。默认不传 `--val_only` 时会处理训练和验证；不传目录参数只是采用默认目录，不表示跳过某个集合。

输入使用 `--train_dir/--val_dir`，输出使用 `--out_train_file/--out_val_file`，两类路径不能混淆。运行结束后应查看写入文档数、跳过数和累计 token 数；当前 `.part` 保护的是输出完整性，不是断点恢复。

训练配置中的 `train_path/dev_path` 应指向实际生成的 HDF5。原版训练入口使用 `config/config.py`，较新的 `pretrain_base.py` 使用其配置加载流程和 `configs/pretrain.json`；预处理完成不会自动修改这些路径。

### 7.4 用小切片检查输出

```python
from pathlib import Path
import h5py

path = Path("/path/to/tokens/pile_train.h5")
with h5py.File(path, "r") as file:
    tokens = file["tokens"]
    print("shape：", tokens.shape, "dtype：", tokens.dtype)
    head = tokens[:100].tolist()
    print("前 100 个 ID：", head)
    assert len(tokens.shape) == 1
    assert tokens.dtype.kind == "i" and tokens.dtype.itemsize == 4
    assert all(0 <= token_id <= 50256 for token_id in head)
```

这个检查只覆盖开头切片，不证明整份数据合法或全量。若解码出现 `�`，先检查是不是恰好截断了字符字节；检查完整文档的往返和检查任意前 100 个 token 是两个不同目标。

## 8. 一套可复用的排查顺序

| 观察到的问题 | 优先核查 |
| --- | --- |
| 相同文字得到不同 ID | tokenizer 文件与版本、规范化、前导空格、预分词、特殊 token 选项 |
| 原文标记被当成结束符 | 正文特殊 token 字面量识别与数据来源协议 |
| 结尾出现两个结束 ID | 数据本来已有结构 EOS，接口是否又自动或手工追加 |
| 显示同样的结束字符串，停止行为不同 | 普通字符序列还是特殊 ID，生成循环是否按 ID 或字符串停止 |
| EOT 后仍有 loss 或跨文档注意力 | 标签右移、attention mask、loss mask 和窗口组织 |
| 输出规模远小于预期 | 是否忘传 `--all_data`、输入是否只有小样本、跳过数与处理范围 |
| 重跑没有接着分词 | `.part` 是输出保护还是实现了输入位置/写入状态恢复 |
| 分词内存增长 | 超长文档、文档批量大小、token 缓冲、完整 JSON 调试导出 |
| 找不到 HDF5 或读取了旧文件 | 预处理输出路径与实际训练配置路径是否一致 |

复现一次数据准备，应记录输入文件、有效记录数、跳过数、tokenizer 定义与版本、正文/结构处理策略、输出类型和累计 token 数。把这些信息与训练样本构造和生成停止条件一起看，才能解释一个分词选择如何影响最终模型行为。

## 参考实现与资料

1. [data_preprocess.py](https://github.com/lankuohsing/train-llm-from-scratch/blob/b6b439fd40cc1505f9695d285113c0cd17151335/scripts/data_preprocess.py)：本地原始文本分词、HDF5 写入与 CLI。
2. [prepare_pretrain_data.py](https://github.com/lankuohsing/train-llm-from-scratch/blob/b6b439fd40cc1505f9695d285113c0cd17151335/scripts/prepare_pretrain_data.py)：下载、批量编码与限量处理。
3. [data_loader.py](https://github.com/lankuohsing/train-llm-from-scratch/blob/b6b439fd40cc1505f9695d285113c0cd17151335/data_loader/data_loader.py)、[attention.py](https://github.com/lankuohsing/train-llm-from-scratch/blob/b6b439fd40cc1505f9695d285113c0cd17151335/src/models/attention.py)、[transformer.py](https://github.com/lankuohsing/train-llm-from-scratch/blob/b6b439fd40cc1505f9695d285113c0cd17151335/src/models/transformer.py)：样本、注意力、loss 与原版生成循环。
4. [evaluation.py](https://github.com/lankuohsing/train-llm-from-scratch/blob/b6b439fd40cc1505f9695d285113c0cd17151335/src/post_training/evaluation.py)、[rollout.py](https://github.com/lankuohsing/train-llm-from-scratch/blob/b6b439fd40cc1505f9695d285113c0cd17151335/src/post_training/rollout.py)：公共生成停止条件。
5. OpenAI. [GPT-2 编码实现](https://github.com/openai/gpt-2/blob/master/src/encoder.py)、[tiktoken 编码定义](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)与 [特殊 token API](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py)。
6. Hugging Face. [GPT-2 tokenizer](https://huggingface.co/docs/transformers/model_doc/gpt2)与 [Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)。
7. Meta Llama. [Llama 3 特殊 token 与消息格式](https://github.com/meta-llama/llama3/blob/main/README.md)、[tokenizer 实现](https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py)。
