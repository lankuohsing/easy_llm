# 大语言模型 Tokenization：方法、演进与实践

理解 tokenizer，可以先回答一个问题：**面对任意文本，我们准备让模型用哪些离散符号表示它，并为此付出多少序列长度、词表开销和信息损失？** BPE、WordPiece、Unigram 都是在回答这个问题，但完整答案还包括文本规范化、切分边界、未知字符处理与特殊 token 约定。

本文先建立一套共同流程，再讨论各环节的设计取舍，用 GPT-2、BERT、Llama 2 和 T5 说明这些取舍如何组合成具体实现。阅读一个新 tokenizer 时，也可以沿用同一套问题。BPE 的逐轮训练、手算与完整教学代码见 [BPE：从训练到编码与解码](<./02_bpe：从训练到编码与解码.md>)。

将这些选择落实到原始语料、EOS/EOT、HDF5、训练样本和生成停止条件的完整实践笔记见 [Pile 分词实践：从原始文本到训练与生成](<./03_pile分词实践：从原始文本到训练与生成.md>)。

## 1. 从表示问题理解 tokenizer

### 1.1 为什么通常选择子词

模型最终接收整数 ID，并通过嵌入矩阵把 ID 转成向量。token 可以代表一个字节、一个字符、一段词、一个完整词或一个结构标记；它不一定是语言学意义上的“词”。不同 tokenizer 即使都包含文本 `hello`，也可能给它不同 ID。**ID 的含义由词表与配套模型共同确定**，编号的数值大小不表示语义相近程度。

最容易想到的三种表示方式，各有代价：

| 表示方式 | 好处 | 主要代价 |
| --- | --- | --- |
| 整词 | 常见句子的序列较短 | 词形、专名、代码与新词不断增加，有限词表很难覆盖 |
| 字符或字节 | 基础单位少，容易组合出新内容 | 常见长字符串也要逐个单位表示，序列通常较长 |
| 子词 | 高频片段用较少 token，低频内容拆成更小单位 | 切分结果依赖语料和规则，词表预算需要分配 |

子词方法的核心思想是**把有限词表预算分配给有用的常见片段，同时保留表示长尾内容的办法**。因此，词表训练语料中的语言比例、代码比例、数字格式与文本来源，都会影响最终的切分。

### 1.2 “更少 token”需要和其他目标一起评估

较短的序列通常减少模型处理的时间步，也能让固定上下文窗口容纳更多文本；更大的词表则增加嵌入层和输出层的开销。如果嵌入维度固定为 $d$ ，一个包含 $|V|$ 个词条的嵌入矩阵需要约 $|V|d$ 个参数。输入嵌入与输出权重可以共享，但大词表仍影响输出投影的计算。

同时，每个独立 token 都需要足够的训练机会。把极少出现的长字符串加入词表，可能减少它的编码长度，却让该 token 的表示缺乏训练。**词表大小、序列长度与词条训练频次要一起考虑**，不能仅凭压缩率判断模型质量。

这个取舍还受任务影响：保留代码缩进、网址或大小写，可能比合并更多常见英文词更有价值；面向多语言的数据，也需要关注每种语言各自得到多少词表覆盖。

## 2. 用一套共同流程理解不同实现

### 2.1 训练、编码与解码共享哪些工作

下面是概念流程。具体库可能把多个步骤合并到一个组件，也可能先识别受保护的特殊 token，再处理普通文本；它不是对所有实现执行顺序的硬性规定 [1]。

```text
训练 tokenizer：
代表性语料 → 固定文本处理与边界策略 → 学习词表及切分参数 → 保存配置

编码普通文本：
原文 → 规范化（可选）→ 边界划分（可选）→ 按已学参数切分 → token ID

组装模型输入：
正文 ID + 任务要求的结构 token → 完整输入 ID、掩码等 → 模型

解码：
ID → 查出对应的文本片段或字节 → 解码器拼接与还原 → 文本
```

聊天模板可以先把消息渲染为带角色和边界的文本，再交给 tokenizer；另一些接口直接组装 ID。两者都需要明确区分正文与结构，不能假定所有 BOS/EOS 都在同一个后处理环节添加。

| 环节 | 各方法共用的职责 | 具体实现可以不同的地方 |
| --- | --- | --- |
| 语料准备 | 决定 tokenizer 看到怎样的数据分布 | 语言配比、去重、领域采样与清洗 |
| 输入表示 | 决定哪些信息进入切分过程 | 大小写、Unicode 规范化、空白、字符或字节表示 |
| 边界划分 | 限制哪些片段可以一起切分或合并 | 按词、标点、字符类别或数字规则分片，也可以不预先按词分片 |
| 学习与切分 | 用有限词表表示输入 | BPE 的合并优先级、WordPiece 的词内匹配、Unigram 的切分分数 |
| ID 映射 | 把最终 token 对接模型的整数输入 | 每个词条的 ID、普通与特殊 token 的分配 |
| 任务包装 | 表达模型需要的角色和序列结构 | BOS/EOS、句对标记、聊天模板、填充与掩码 |
| 解码 | 从 ID 重建可展示的内容 | 字节拼接、空格还原、续接标记处理、是否跳过特殊 token |

### 2.2 哪些需要学习，哪些是事先约定

词表和切分参数通常从语料学习；是否转小写、如何分片、如何处理未知字符，以及哪些 ID 是 EOS，通常由设计者选择或配置。训练 tokenizer 时，两部分一起决定产物。

训练语言模型则是另一个任务：使用这些 ID 学习下一 token 或其他训练目标。常规编码新文本时，tokenizer 不再更新词表，也不重新统计当前文本的频次。Unigram 切分采样、BPE dropout 等方法可以显式引入切分随机性，但这不等于每次编码都重新训练 tokenizer。

解码通常只需要反向词表和解码器，**不需要重新运行切分算法，也不需要倒放 BPE 合并过程**。如果编码前丢失了大小写、空白或未知字符信息，解码器也无法把它凭空恢复。

### 2.3 算法、框架、编码配置属于不同层次

| 名称 | 所在层次 | 它告诉我们什么 |
| --- | --- | --- |
| BPE、WordPiece、Unigram | 学习与切分方法 | 如何获得词条、如何选择切分 |
| 字节级 BPE | BPE 的输入表示选择 | 基础符号是字节，仍然使用 BPE 合并思想 |
| SentencePiece | 分词训练与编码框架 | 可运行 BPE 或 Unigram，并保存文本处理等配置 |
| Hugging Face Tokenizers / Transformers | 实现库及模型接口 | 将规范化、预分词、切分、解码和模型包装连接起来 |
| tiktoken | 编码实现库 | 使用字节串与 rank 等配置实现 BPE，支持内置和自定义编码 |
| `r50k_base`、`cl100k_base` | 具体编码配置 | 固定的词表、预分词规则和特殊 token 定义 |

因此，“使用 SentencePiece”还不足以确定是 BPE 还是 Unigram；“使用 BPE”也不足以确定字符覆盖、空格处理或 token ID。

## 3. 设计 tokenizer 时，需要独立考虑的五个问题

### 3.1 信息保留：哪些变化可以视为同一种文本

规范化决定 tokenizer 是否区分某些输入。例如转小写后，`Hello` 与 `hello` 可以复用表示；去除重音后，`Café` 可以变成 `cafe`。这能集中统计，但也会丢掉原文差异。

对于代码，`UserID` 与 `userid` 可能是不同标识符；对于原始语料分析，两个空格与一个空格也可能需要区分。Unicode NFKC 还可能把兼容字符转换成其他形式。这些都应作为任务选择，而不是无条件执行的“清洗”。

判断方式很具体：**先写出哪些原文差异必须保留，再选择规范化规则。** 若采用不可逆规则，应把往返目标定义为“恢复规范化后的文本”，而不是任意原文逐字节一致。

### 3.2 边界约束：哪些内容允许形成一个 token

预分词为后面的算法划定活动范围。BPE 只能在允许的片段内部合并；如果先把标点、数字或词边界切开，再多的合并规则也不能跨过这些边界。

例如，同样面对 `1234567`：

| 边界策略 | 切分算法看到的片段 | 后果 |
| --- | --- | --- |
| 保留完整数字串 | `1234567` | 允许学习覆盖较长数字串的 token |
| 每次只允许一个数字 | `1 / 2 / 3 / 4 / 5 / 6 / 7` | 位数表示统一，序列较长 |
| 从左到右按最多三位划分 | `123 / 456 / 7` | 限制数字词条长度，算法无法跨组组合 |

GPT-2 的预分词允许连续数字处于一个片段中；`cl100k_base` 和 `o200k_base` 的数字分支采用 1–3 位的规则 [3]。**这是预分词边界，不等于最终 token 切分**：一个三位数字片段仍可能被 BPE 拆成多个 token。

边界带来的好处是切分更可控，也可能减少低价值组合；代价是禁止了某些压缩机会。不同的数字边界还改变模型学习位值和计算模式的条件，不能仅凭“更规整”断言算术效果更好。

### 3.3 基础单位与回退：如何表示训练时没有见过的内容

| 策略 | 常见片段如何处理 | 未覆盖内容如何处理 | 典型代价 |
| --- | --- | --- | --- |
| 有限字符或子词词表 + 未知 token | 使用已有字符或词条 | 映射为 `[UNK]` 或 `<unk>` | 原文可能丢失，未知规则因实现而异 |
| 字节级 BPE | 从 256 个基础字节出发，再合并常见字节串 | 所有有效 UTF-8 文本可由基础字节表示 | 未充分覆盖的语言可能产生较长序列 |
| 字符/子词 + byte fallback | 优先使用已有字符或子词 | 未覆盖字符回退为 UTF-8 字节 token | 回退序列可能较长，需要配套字节解码 |

GPT-2 属于第二种，Llama 2 的 SentencePiece BPE 使用第三种。**“从字节开始做 BPE”与“未知字符才回退到字节”是两种不同的配置**，不能因为都能看到字节 token 就认为整个流程相同。

字节覆盖保证可表示性，词表中的合并覆盖决定表示效率。同一个中文字符可以最终成为一个 token，也可能分散在多个 token 中；字节数不等于最终 token 数。

### 3.4 切分准则：怎样分配词表预算、选择一条编码路径

| 方法 | 训练时的主要思想 | 保存的核心信息 | 编码时的选择方式 |
| --- | --- | --- | --- |
| BPE | 从小符号开始，反复合并当前高频相邻对 | 词表、有序合并规则或 rank | 按合并优先级组合当前相邻片段 |
| WordPiece | 学习适合表示词内片段的词表；公开教程常用相对共现打分解释训练 | 词表与续接约定 | 对每个预分词词，从左到右做最长匹配 |
| Unigram | 从较大候选词表出发，估计概率并逐步删除贡献较小的词条 | 词表、词条分数或概率 | 比较合法切分的整体分数，通常用动态规划求最优路径 |

BPE 偏向保留反复出现的组合；WordPiece 的编码器直接依赖词表匹配；Unigram 显式比较多条候选切分路径。这些差别影响词表训练和编码，**不改变“最终 token 映射为 ID”这一共同接口**。

WordPiece 原版训练实现未由 Google 完整公开。Hugging Face 教程使用的“pair 频次除以两个片段频次之积”应理解为教学复现准则，不能直接当作已验证的原版训练公式 [4]。BPE 与 WordPiece 的具体训练演算，也不能只看最终词表就互相替代。

### 3.5 结构协议：正文和控制标记如何区分

特殊 token 让有限的整数序列表达额外结构，例如文档结束、句对边界、聊天角色或待补全片段。它们通常是手工定义的协议，不能仅凭名字推断模型会执行什么动作；模型需要在相应训练数据中学习其用途。

例如，Pile 的 JSONL `text` 字段可能引用字符串 `<|endoftext|>` 来讲解 GPT-2。合理的数据处理约定是将正文中的字面量按普通文字编码，在一条真实文档结束后再追加 ID `50256`。如果格式明确规定正文中的这串字符就是分隔符，则应按该格式解析。

这说明两个需要独立检查的设置：**是否识别正文中的特殊 token 字面量，以及是否额外添加 BOS/EOS。** Hugging Face 的 `add_special_tokens=False` 控制后者，并不自动关闭前者。结束 token 也不会自动阻断不同文档之间的注意力；掩码和训练样本组织需要另行设置。

## 4. 四个典型实现：沿同一套流程比较

### 4.1 GPT-2 / `r50k_base`：可逆字节表示 + 正则边界 + BPE

GPT-2 的设计组合是：保留大小写等普通文本内容，以 UTF-8 字节为基础，用正则限制合并边界，再根据 BPE 规则压缩常见字节串 [2][3]。

**训练得到什么。** 256 个基础字节 token 加上 50,000 次合并产生的普通 token，再加入一个 `<|endoftext|>`，共 50,257 个 ID。GPT-2 风格文件保存内部字符串词表和合并对；`r50k_base` 保存字节串到 rank/ID 的表，另有特殊 token 定义。

**如何编码。** 对真实文本 `" the"`，前导空格与后面的字母可以处于同一预分词片段。空格字节在 GPT-2 内部映射中显示为 `Ġ`，实际合并如下：

```text
原文：       " the"
初始符号：   Ġ t h e
合并：       Ġ t h e → Ġt h e → Ġt he → Ġthe
最终 ID：    [262]
```

| `model.merges` 中的规则 | rank：数组下标 | 合并结果在 `model.vocab` 中的 ID |
| --- | --- | --- |
| `Ġ t` | 0 | `Ġt` → 256 |
| `h e` | 2 | `he` → 258 |
| `Ġt he` | 6 | `Ġthe` → 262 |

数组下标越小，规则优先级越高。原版 GPT-2 的合并结果 ID 恰好等于规则下标加 256；这个编号关系不能推广到全部 BPE。tiktoken 的普通 rank 同时承担 ID 的职责，因此 `b" the"` 的 rank/ID 是 `262`，而不是 GPT-2 合并表的下标 `6`。

`Ġ` 和表示换行的 `Ċ` 是内部字节映射字符。原文中真正出现的 Unicode 字符 `Ġ` 会按自己的 UTF-8 字节编码，不会被替换为空格。

**如何解码。** 查出每个 ID 对应的字节串，全部连接，再做 UTF-8 解码。单个 token 可能只包含中文字符的一部分字节，所以逐 token 解码后拼字符串不一定正确。普通有效文本在选项一致、没有截断或其他信息丢失时可以完整往返。

**案例说明的取舍。** 字节基础解决覆盖问题，合并词表解决长度问题，正则解决边界约束。三层都要看：编码中文效率低可能来自语料和合并覆盖，不能归结为“字节级 BPE 不能处理中文”。

### 4.2 BERT uncased：规范化 + 词内 WordPiece + 任务标记

BERT uncased 会处理大小写和重音，并按空白、标点等规则形成词级片段；WordPiece 再在每个词内部做从左到右的最长匹配 [4][5]。`##` 表示该词内部的续接片段，不是原文中的两个井号。

下面使用**自定义教学词表**，说明 BERT 风格流程，不是 BERT 真实词表或真实模型 ID：

```text
词表：[UNK]=0，cafe=1，play=2，##ing=3，!=4

原文：              "Café  playing!"
规范化与预分词：    [cafe, playing, !]
WordPiece 切分：    [cafe, play, ##ing, !]
ID：                [1, 2, 3, 4]
decode：            "cafe playing!"
```

**训练得到什么。** 编码依赖 WordPiece 词表和词内续接约定；它不需要 GPT-2 那样的有序合并表。实际训练得到的词表与规范化、预分词规则共同决定覆盖。

**如何编码。** `playing` 先匹配词首 `play`，再匹配续接 `##ing`。在这个教学词表中，`playingly` 的剩余 `ly` 无法匹配，BERT 风格实现会把整个词变成 `[UNK]`。

**如何解码。** 查询 token，处理 `##` 续接并重建词间间隔。原始重音、大小写和两个空格已经在前面的处理里丢失，因此输出不能恢复为原文。信息损失发生在规范化、边界处理与未知词策略，不能归因于“ID 不能还原文字”。

**任务包装有什么特色。** BERT 单句输入常添加 `[CLS]` 与 `[SEP]`，句对输入还使用分段约定和 `token_type_ids`。这些属于模型接口和训练任务；WordPiece 本身并不要求所有使用它的模型都采用 BERT 包装。

**案例说明的取舍。** 复用规范化后的词内片段可以集中表示，但失去了部分表面形式；适合某些任务的约定，未必适合要求精确保留原文的生成式训练。

### 4.3 Llama 2：SentencePiece BPE + 空格表示 + 字节回退

Llama 2 延续了 LLaMA 的 SentencePiece BPE 路线 [6][7]。它与 GPT-2 共享 BPE 合并思想与 ID 接口，但采用另一套文本表示、边界配置和未知字符处理。

**训练得到什么。** SentencePiece 模型保存词条、BPE 所用的分数/优先级和文本处理配置等信息。它可以用一个 `.model` 文件保存多种配置，不要求一定存在单独的 `merges.txt`。

**如何编码。** 空格通过 `▁` 一类内部标记进入表示，模型可配置虚拟前导空格。下面只示意内部表示，省略可能的 BOS，不宣称真实词条或 ID：

```text
文本：                  "Hello world"
空格表示（带前导标记）： ▁Hello▁world
后续：                  按模型的 BPE 优先级得到词条 → 查 ID
```

`▁` 让空格参与模型表示，但“显示了空格标记”不等于“按空格强制分片”。是否允许跨空格组成词条，还取决于训练模型中的边界选项。原版 Llama 2 延续的数字策略是逐位划分，例如 `1234567` 的数字各自处理，BPE 不把它们合并成多位数字 token；这与 GPT-2 的完整数字片段、`cl100k_base` 的最多三位分组形成对照。

遇到没有被字符/子词词表覆盖的字符时，byte fallback 使用其 UTF-8 字节 token。常见输入优先走已有字符或子词，和 GPT-2 从全部基础字节开始合并的过程有区别。

**如何解码。** 拼接词条，还原空格标记、处理前导标记与字节回退，再得到文本。原版 Llama 2 不采用 SentencePiece 默认的 NFKC 字符规范化；空白和前导标记的处理仍需查看具体模型与接口选项，不能仅据此保证任意原文逐字节往返。**SentencePiece 的默认配置不能直接套到 Llama 2** [8]。

**案例说明的取舍。** 框架相同也可以选择不同算法和文本规则；BPE 相同也可以选择不同基础表示。Llama 2 是区分“框架、算法、回退策略”这三层的典型例子。

### 4.4 T5：SentencePiece Unigram + 全局切分分数 + span 标记

经典 T5 使用 SentencePiece Unigram tokenizer [9]。它与 Llama 2 共享框架和部分空格表示机制，切分核心却改为概率模型。

**训练得到什么。** Unigram 从较大的候选词表出发，估计词条概率，逐步删去移除后对语料似然影响较小的候选。保存产物包含词条分数，常以对数形式存储；编码不使用 BPE 那样的合并顺序 [10]。

**如何编码。** 同一字符串可以有多种合法切分，Unigram 用各 token 概率的乘积比较路径，常规编码用动态规划寻找整体分数最高的路径。下面是省略空格表示的**教学概率例子**，不是 T5 实际词表：

```text
p(abc)=0.04，p(a)=0.20，p(bc)=0.30
p(ab)=0.10，p(c)=0.10，p(b)=0.05
其余概率质量分配给不影响本例的其他词条。
```

| `abc` 的候选切分 | 路径概率 |
| --- | --- |
| `[abc]` | `0.04` |
| `[a, bc]` | `0.20 × 0.30 = 0.06` |
| `[ab, c]` | `0.10 × 0.10 = 0.01` |
| `[a, b, c]` | `0.20 × 0.05 × 0.10 = 0.001` |

这时会选择 `[a, bc]`，即使词表里有完整的 `abc`。这个例子说明：Unigram 编码不等于最长匹配，也不一定选择 token 数最少的路径。启用切分采样时，可以从候选路径中抽样，给模型提供不同的切分视图。

**如何解码。** 仍然是查词条、拼接、还原空格等表示，不需要再找最优路径。原版 T5 tokenizer 没有启用 byte fallback，未覆盖的字符可能落入 `<unk>`；其规范化、空白处理和未知字符替换也可能丢失原文信息。这与 Llama 2 的字节回退形成直接对照。T5 衍生模型应按自己的文件核查，不能由框架名称推断行为。

**任务包装有什么特色。** T5 的 `</s>` 和 `<extra_id_0>` 等 sentinel token 服务于序列结束、span corruption 等任务。例如遮住一个文本片段时，输入与目标中的 sentinel 标记建立待预测片段的对应关系。这是训练任务设计，Unigram 也可以用于不采用 sentinel 的模型。

**案例说明的取舍。** 子词预算可以通过概率与剪枝来分配；输出 ID 的共同接口不变，选择切分的决策方式改变了。

### 4.5 将四个案例放回共同流程

| 流程环节 | GPT-2 / `r50k_base` | BERT uncased | Llama 2 | 经典 T5 |
| --- | --- | --- | --- | --- |
| 输入处理 | 可逆字节表示，保留普通文本形式 | 小写、重音等规范化 | 模型规定的空格与文本处理，不能套默认配置 | 模型规定的规范化与空格处理 |
| 边界来源 | 字符类别等正则分片 | 词、标点等预分词边界 | 模型的边界配置，数字逐位划分 | SentencePiece 模型的边界配置 |
| 切分核心 | BPE 合并优先级 | 词内最长匹配 | BPE 优先级 | Unigram 路径分数 |
| 未覆盖输入 | 基础字节可表示 | 可能整个词变为 `[UNK]` | 字节回退 | 未覆盖字符可能为 `<unk>`，未启用字节回退 |
| 映射接口 | token → ID | token → ID | token → ID | token → ID |
| 解码 | 字节拼接后 UTF-8 解码 | 处理续接标记并重建间隔 | 还原空格与回退字节 | 还原词条与空格表示 |
| 结构约定举例 | `<\|endoftext\|>` | `[CLS]`、`[SEP]` | BOS/EOS，聊天版另有模板 | `</s>`、sentinel token |

这张表提供的是核查方向。具体文件与版本才确定某个实例的精确行为，例如 BERT cased 与 uncased、基础版与聊天版不能直接混用配置。

## 5. 从模型演进理解设计变化

理解演进时，可以逐项追问：**基础单位变了吗，边界变了吗，词表预算怎么分配，模型接口新增了什么结构？** 模型发布日期和总词条数只提供部分信息。

| 典型变化 | 共用的部分 | 改变的部分 | 学习这个例子的目的 |
| --- | --- | --- | --- |
| GPT-2 → GPT-3 | 延续 GPT-2 字节级 BPE | 语言模型规模与训练等方面变化 | 模型升级不要求 tokenizer 同时更换 [11] |
| `r50k_base` → `cl100k_base` / `o200k_base` | 字节级 BPE 与整数接口 | 词表、数字/字母/空白等预分词规则、特殊 token 配置 | 同样是 BPE，也需要逐层比较 [3] |
| Llama 2 → Llama 3 | 子词输入与生成式建模接口 | SentencePiece BPE 路线转向 tiktoken 风格的字节级 BPE，词表与边界等随之改变 | 同一模型家族不能直接沿用旧 tokenizer [12] |
| Llama 2 与 T5 | 都使用 SentencePiece | BPE 与 Unigram、规范化/回退及任务包装不同 | 工具框架不能代替算法与配置的描述 |

多语言词表的变化，应看各语言的词条覆盖和序列长度。例如增加中文常见字节串的合并机会，可以在保留字节基础的同时缩短中文编码。更大的总词表并不自动保证每一种语言都获益；必须在相同语料、相同统计口径下测量 [13]。

也不存在所有模型按年份单调扩表的共同路径。例如 BLOOM 较早就采用了约 250K 的多语言词表 [14]。取舍由语料和模型预算决定，不能把词表规模当作技术代际排名。

BLT、H-Net 等研究把固定子词边界进一步替换为字节输入和动态或可学习的分块 [15][16]。它们继续面对“表示长度与计算量如何分配”的问题，只是将一部分边界决策移入模型。是否在特定规模更有效，需要比较实际训练与部署结果，不能仅由字节序列变长推导一个通用计算倍数。

## 6. 选择与评估：把设计问题变成可验证的步骤

### 6.1 使用已有模型：先保证协议一致

使用已有模型时，应先加载与权重配套的 tokenizer 和模板，再考虑吞吐优化。同样的可见文本必须按模型训练时的规则编码，不能仅因为另一个词表更短就直接替换。

增加词条需要调整嵌入和输出映射，并训练新词条的表示；整体更换 tokenizer 还改变切分及 ID 含义，属于需要验证和重新训练适配的模型改造。若只是从 tiktoken 换到等价 Hugging Face 实现，可以通过词表、规则和 API 行为检查来验证；这是实现替换，条件是模型实际收到的 ID 一致。

### 6.2 从头训练：先明确数据与约束，再选算法

1. **定义需要保留的信息。** 明确是否要求原文往返、大小写和缩进保留，以及正文如何处理特殊 token 字面量。
2. **构造有代表性的 tokenizer 训练语料。** 覆盖目标语言、代码、数字、网址等类型，并记录采样配比；从一开始就保留独立验证语料。
3. **固定候选的表示和边界策略。** 字符或字节、未知回退、数字分片等应显式配置，避免将多个变化误归因于一种算法。
4. **在相同数据上训练候选词表。** 比较几个可承受的词表预算，保存全部配置、版本与训练信息。
5. **按语言和文本类型测量编码行为。** 同时观察覆盖、长度分布、词条频次、可逆性与运行速度。
6. **通过模型实验验证。** 在明确的训练数据与计算预算下比较任务效果、吞吐和部署成本；tokenizer 的压缩率不能代替这一步。

### 6.3 测量时避免混淆什么

| 指标或实验 | 建议做法 | 它回答的问题 |
| --- | --- | --- |
| 字符或字节覆盖 | 统计未知 token、字节回退及往返失败，检查失败原文 | 能否保留需要的信息 |
| 表示长度 | 分语言/类型报告每字符、每 UTF-8 字节的 token 数，以及文档长度分布 | 固定窗口能放多少内容，长尾样本有多长 |
| 词条使用频次 | 看低频和未使用词条的比例，区分普通与结构 token | 词表预算是否分配给了会被充分训练的内容 |
| 编码性能 | 在相同机器、批量与文本上测量速度和内存 | 数据处理或服务的实际成本 |
| 模型效果 | 固定并记录数据与计算口径，比较任务结果 | 表示变化最终是否帮助模型 |

比较两个 tokenizer 的压缩率时，应统一是否计入特殊 token，使用同一批原文，并单独报告各语言和文本类型。平均值相同也可能掩盖某类文本的长度恶化。

比较语言模型时，两个 tokenizer 的“平均每 token loss”不能直接横比，因为它们的 token 单位不同。需要对齐文本与预处理、明确按原始字节或其他统一单位统计的似然指标，并结合任务评测。相同 token 数的训练预算也可能对应不同的原始文本量，应在实验记录中说明。

如果规范化或未知 token 替换已经丢失了原文信息，即使按原始字节数归一化 loss，也不能单独判断对原文的建模质量。应同时报告覆盖和往返结果，确保指标没有掩盖被丢弃的内容。

## 7. 实践核查：沿流程定位差异

### 7.1 配置文件应映射到哪些环节

读取文件时，先确定对应环节，再看字段。对 Hugging Face Tokenizers，完整流程通常在 `tokenizer.json` 中；其他实现的文件组织不同 [1][17]。

| 环节 | 常见保存位置 | 重点检查 |
| --- | --- | --- |
| 规范化 | `tokenizer.json` 的 `normalizer`，或 SentencePiece 模型中的规范化配置 | 是否改变大小写、Unicode 或空白 |
| 预分词 | `pre_tokenizer`、tiktoken 的 `pat_str`、SentencePiece 的模型配置 | 哪些边界限制后续切分 |
| 学习参数 | `model.vocab`、`model.merges`、WordPiece 词表、Unigram 词条分数 | ID、规则次序或概率是否与目标模型一致 |
| 独立词表/规则 | `vocab.json`、`merges.txt`、`vocab.txt` 等 | 词表文件只保存其中一部分信息，不能一概认为它是完整 tokenizer |
| 结构与包装 | `added_tokens`、`post_processor`、`tokenizer_config.json`、特殊 token 映射及聊天模板 | 字符串、ID、角色，以及是否自动添加 |
| 解码 | `decoder`、SentencePiece 模型及接口选项 | 字节、空格、续接标记与输出清理 |

SentencePiece `.model` 可以包含算法、词条分数和规范化等信息；不要因没有 `merges.txt` 就认定不存在 BPE 优先级。`tokenizer_config.json` 是包装层配置，可能保存 tokenizer 类、特殊 token 角色和长度设置，通常不保存合并表。不同版本的字段与文件组合可能不同。

GPT-2 的 `tokenizer.json` 可能以紧凑格式保存为一整行。下面直接读取关键字段，只依赖标准库，将路径替换为本地目录即可：

```python
import json
from pathlib import Path

tokenizer_dir = Path("/path/to/gpt2")
config = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
print("规范化：", config.get("normalizer"))
print("预分词：", config.get("pre_tokenizer"))
print("词条数：", len(config["model"]["vocab"]))
print("前七条合并规则：", config["model"]["merges"][:7])
print("解码器：", config.get("decoder"))
```

有些版本把合并对保存为 `"h e"`，有些保存为 `["h", "e"]`，含义相同。比较不同表示时，应先统一字节/字符串与规则含义。

### 7.2 验证实现等价：从词表到 API 行为

GPT-2 与 `r50k_base` 使用相同的普通词表和合并优先级，适合验证“同一算法配置、不同实现”的情况 [2][3]。比较应覆盖四层：普通 token 的字节内容与 ID、合并优先级、预分词与规范化、特殊 token 及包装选项。只抽查一句英文，或者只比较总词条数，证据不够。

对于正文中的 `"Hello <|endoftext|> world"`，tiktoken 的 `encode_ordinary` 按普通文字编码该字面量；Hugging Face GPT-2 默认会识别它为特殊 token。下面显式对齐正文策略，再在真实文档结尾添加 EOT：

```python
import tiktoken
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained(
    "/path/to/gpt2",
    local_files_only=True,
    add_prefix_space=False,
)
encoding = tiktoken.get_encoding("r50k_base")

samples = [" the", "你好，世界！", "a  b\n", "Hello <|endoftext|> world"]
for text in samples:
    hf_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        split_special_tokens=True,
    )
    token_ids = encoding.encode_ordinary(text)
    assert hf_ids == token_ids
    assert encoding.decode(token_ids) == text
    print(repr(text), token_ids)

    # 正文编码结束后，再添加文档结束标记。
    assert tokenizer.eos_token_id == encoding.eot_token == 50256
    hf_ids.append(tokenizer.eos_token_id)
    token_ids.append(encoding.eot_token)
    assert hf_ids == token_ids
```

需要安装 `tiktoken` 与 `transformers`，准备本地原版 GPT-2 tokenizer 文件；tiktoken 首次加载可能需要联网下载词表。此例在 `transformers 5.3.0`、`tokenizers 0.22.2`、`tiktoken 0.12.0` 下验证。显式使用 GPT-2 类，避免不完整配置被加载为通用后端而没有设置 EOS 角色；更换文件或版本后仍应检查字段与实际 ID。

配置中的 `model_max_length=1024` 是包装层长度信息，默认编码不会因此自动截断。截断、填充、正文特殊 token 识别和额外边界 token 都是需要显式核查的行为。

### 7.3 用诊断文本定位问题发生在哪一步

| 诊断输入 | 优先检查的环节 |
| --- | --- |
| 大小写、重音、全角字符 | 规范化是否改变内容 |
| 连续空格、缩进、换行、尾随空白 | 空白保留、预分词与解码清理 |
| 长数字、带符号的数值、代码标识符 | 数字/标点边界与合并覆盖 |
| 中文、生僻字、emoji、组合字符 | 词表覆盖、未知回退与 UTF-8 字节拼接 |
| `<\|endoftext\|>` 或聊天角色标记字面量 | 正文与结构的识别约定 |
| 相同正文经不同库加载 | 自动 BOS/EOS、前导空格、版本与 ID 映射 |

排查时先比较处理后的文本与片段边界，再比较 token 内容和 ID，最后检查模型包装和解码选项。对已有模型，应同时保存 tokenizer 文件、模板与版本，使训练、评测和推理能复现同一套输入协议。

## 参考资料

1. Hugging Face. [Tokenizers pipeline](https://huggingface.co/docs/tokenizers/main/en/pipeline)，各组件的职责与执行流程。
2. OpenAI. [GPT-2 编码器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py)；Hugging Face. [GPT-2 tokenizer.json](https://huggingface.co/openai-community/gpt2/blob/main/tokenizer.json) 与 [merges.txt](https://huggingface.co/openai-community/gpt2/blob/main/merges.txt)。
3. OpenAI. [tiktoken 编码定义](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)、[encode / decode 与特殊 token 实现](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py)及 [自定义编码](https://github.com/openai/tiktoken#extending-tiktoken)。
4. Hugging Face. [WordPiece tokenization](https://huggingface.co/learn/llm-course/en/chapter6/6)，包含训练公开程度与教学复现说明。
5. Google Research. [BERT BasicTokenizer 与 WordpieceTokenizer](https://github.com/google-research/bert/blob/master/tokenization.py)；Hugging Face. [BERT tokenizer](https://huggingface.co/docs/transformers/model_doc/bert)。
6. Touvron et al. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971). 2023.
7. Touvron et al. [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288). 2023.
8. SentencePiece. [官方项目](https://github.com/google/sentencepiece)、[规范化说明](https://github.com/google/sentencepiece/blob/master/doc/normalization.md)与 [模型配置字段](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto)。
9. Raffel et al. [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html). JMLR, 2020；Hugging Face. [T5 tokenizer](https://huggingface.co/docs/transformers/model_doc/t5)。
10. Kudo. [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://aclanthology.org/P18-1007/). ACL, 2018；Hugging Face. [Unigram tokenization](https://huggingface.co/learn/llm-course/en/chapter6/7)。
11. Brown et al. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). NeurIPS, 2020.
12. Llama Team. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). 2024.
13. Petrov et al. [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425). NeurIPS, 2023.
14. BigScience Workshop. [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100). 2022.
15. Pagnoni et al. [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871). 2024.
16. [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955). 2025.
17. Hugging Face. [Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)、[GPT-2 tokenizer](https://huggingface.co/docs/transformers/model_doc/gpt2)及 [添加词条与调整嵌入](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings)。
