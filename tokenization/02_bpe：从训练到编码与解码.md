# BPE：从训练到编码与解码

BPE（Byte Pair Encoding）通过反复合并语料中高频的相邻符号，学习一套子词词表和有序合并规则。使用这套规则时，`encode` 将文本切成 token 并映射为整数 ID；`decode` 查出 ID 对应的内容，再还原文本。本文用一个可以手算的字符级例子贯穿这三个过程，最后对应到 GPT-2 的字节级 BPE 和 `r50k_base`。

主流分词方法与配置文件的读法见 [大语言模型 Tokenization：方法、演进与实践](<./01_大语言模型 Tokenization：方法、演进与实践.md>)。本文的教学例子使用自行分配的 ID，不能拿这些 ID 直接输入 GPT-2。

## 1. 三个过程分别做什么

这里的“训练 BPE tokenizer”指从文本统计中学习切分规则，不涉及 Transformer 的参数、反向传播或下一 token 预测损失。训练语言模型时，通常先使用已经确定的 tokenizer 将语料转换为 ID，再学习这些 ID 序列的分布。

| 过程 | 输入 | 主要工作 | 输出 |
| --- | --- | --- | --- |
| 训练 tokenizer | 代表性的文本语料、基础符号、合并预算等配置 | 统计相邻符号对，逐轮合并并记录次序 | 词表、合并规则，以及配套处理配置 |
| `encode` | 一段文本、已训练的 tokenizer | 预分词、按固定规则合并、查 token ID | 整数序列 |
| `decode` | 整数序列、同一个 tokenizer | 查 ID 对应的文本或字节，拼接并还原 | 文本 |

可以把数据流写成：

```text
训练：语料 → 基础符号序列 → 反复统计与合并 → 词表 + 有序规则

编码：文本 → 预分词 → 基础符号 → 按已学规则合并 → token → ID

解码：ID → token 对应的文本或字节 → 按顺序拼接 → 文本
```

训练时决定规则，编码时使用规则。**编码一条新文本时，不会根据这条文本的频次重新训练词表。**

## 2. 先确定基础符号和文本边界

### 2.1 基础符号是什么

字符级 BPE 可以从语料中出现的字符开始。例如 `hug` 最初表示为三个符号：

```text
[h, u, g]
```

GPT-2 使用字节级 BPE，基础符号是 256 个字节值。对普通有效 Unicode 文本，先转换为 UTF-8 字节，再做 BPE。因此它可以表示训练时没有见过的字符，而不需要因为字符缺失而使用未知 token。字符级小词表若没有包含某个字符，就必须另行处理未知字符。

无论基础单位是字符还是字节，**一次合并都把两个当前相邻符号变成一个新符号**。这个新符号可以继续参与后面的合并。

### 2.2 在哪里允许合并

实际 tokenizer 通常先做预分词，确定允许 BPE 合并的片段边界。本文使用简单约定：将连续非空白内容和连续空白分别保留为片段，不丢弃空格；BPE 只在每个片段内部工作。

```text
"hug bugs" → ["hug", " ", "bugs"]
```

这与直接调用 `text.split()` 有区别：`split()` 默认会丢弃空白及其数量，之后就无法准确恢复原文中的连续空格。GPT-2 的预分词规则更复杂，也允许一个前导空格与后面的词处于同一片段；本文的边界约定是为了便于演示。

如果 tokenizer 还配置了小写转换、Unicode 归一化等操作，训练和编码必须采用同样的处理方式。BPE 只能保留传给它的信息，无法还原前面已经丢弃的内容。

## 3. 训练：从频次统计得到合并规则

### 3.1 准备一个小语料

假设完成预分词后，语料包含下面四种词。将相同片段压缩为“片段 + 出现次数”，可以避免重复处理相同数据：

| 片段 | 在语料中的出现次数 | 初始符号序列 |
| --- | --- | --- |
| `hug` | 10 | `h u g` |
| `hugs` | 5 | `h u g s` |
| `bug` | 6 | `b u g` |
| `bugs` | 2 | `b u g s` |

基础字符为 `b、g、h、s、u`。为了后面演示整句编码，另外显式预留空格。本文按下表分配初始 ID；编号是人为约定，不是字符的 Unicode 编码，也不是算法计算出的语义数值。

| token 内容 | 初始 ID |
| --- | --- |
| 空格 `" "` | 0 |
| `b` | 1 |
| `g` | 2 |
| `h` | 3 |
| `s` | 4 |
| `u` | 5 |

### 3.2 第一轮：统计当前相邻对

每个片段内部，相邻两个符号组成一个候选 pair。它的总频次要乘以片段在语料中的出现次数。

| 相邻对 | 频次来源 | 总频次 |
| --- | --- | --- |
| `h + u` | `hug` 的 10 次 + `hugs` 的 5 次 | 15 |
| `u + g` | 四种词的 10 + 5 + 6 + 2 次 | 23 |
| `b + u` | `bug` 的 6 次 + `bugs` 的 2 次 | 8 |
| `g + s` | `hugs` 的 5 次 + `bugs` 的 2 次 | 7 |

最高频的是 `u + g`，所以第一条规则为：

```text
规则 0：u + g → ug
新 token：ug，ID = 6
```

把这对符号在各片段中的出现位置合并后，语料变为：

```text
hug   × 10：h ug
hugs  ×  5：h ug s
bug   ×  6：b ug
bugs  ×  2：b ug s
```

**合并后必须重新统计。** 现在已经没有 `u + g` 这个相邻对，而是出现了 `h + ug`、`b + ug` 和 `ug + s`。

### 3.3 继续训练：新符号参与下一轮合并

假设合并预算是 4 次，完整训练过程如下。表中的状态都保留原片段的出现次数。

| 轮次 | 当前候选对及加权频次 | 本轮选择 | 新 ID | 合并后的四种片段 |
| --- | --- | --- | --- | --- |
| 1 | `h+u:15`，`u+g:23`，`b+u:8`，`g+s:7` | `u + g → ug` | 6 | `h ug`；`h ug s`；`b ug`；`b ug s` |
| 2 | `h+ug:15`，`b+ug:8`，`ug+s:7` | `h + ug → hug` | 7 | `hug`；`hug s`；`b ug`；`b ug s` |
| 3 | `hug+s:5`，`b+ug:8`，`ug+s:2` | `b + ug → bug` | 8 | `hug`；`hug s`；`bug`；`bug s` |
| 4 | `hug+s:5`，`bug+s:2` | `hug + s → hugs` | 9 | `hug`；`hugs`；`bug`；`bug s` |

达到预算后停止。`bugs` 仍然是 `[bug, s]`，因为 `bug + s` 尚未成为合并规则。**训练不会要求所有词都变成单个 token**；常见片段有机会获得独立 token，其他内容可以由多个 token 组成。

实际训练还可能使用目标词表大小、最低 pair 频次等停止条件。若最高频 pair 并列，需要确定的选择策略才能复现训练结果，例如按 pair 的字典序选择。本例各轮最高频 pair 都没有并列。

统计与替换还要区分重叠情况：`aaa` 中有两个相邻位置匹配 `a + a`，但一次从左到右的非重叠合并会得到 `[aa, a]`，不能把共享的中间字符使用两次。后面的示例代码采用“统计所有相邻位置、执行非重叠合并”的约定。

### 3.4 最终保存什么

训练得到的词表包含初始符号和历次新增 token，不能只保留最终语料里还出现的片段：

```text
vocab = {
    " ": 0, "b": 1, "g": 2, "h": 3, "s": 4, "u": 5,
    "ug": 6, "hug": 7, "bug": 8, "hugs": 9
}
```

规则必须保留顺序，通常用 rank 表示优先级：

| rank，从 0 开始 | 左片段 | 右片段 | 合并结果 |
| --- | --- | --- | --- |
| 0 | `u` | `g` | `ug` |
| 1 | `h` | `ug` | `hug` |
| 2 | `b` | `ug` | `bug` |
| 3 | `hug` | `s` | `hugs` |

这里 `vocab` 告诉我们“片段对应哪个 ID”，规则告诉我们“两个相邻片段能否合并、谁先合并”。rank 与 token ID 是不同职责；本例恰好按新增顺序分配 ID，也可以给相同 token 分配别的 ID。

完整 tokenizer 还应保存预分词、归一化、基础符号表示、特殊 token 和解码方式。只保存词表大小或一个普通词表文件，通常不足以复现编码行为。

## 4. Encode：用固定规则把文本变成 ID

### 4.1 编码一个训练中没有出现的完整字符串

使用刚训练好的规则编码 `uhugs`。这个完整字符串不在训练语料中，但它的字符都在基础词表中。

```text
初始：     u h u g s
规则 0：   u h ug s       u + g → ug
规则 1：   u hug s        h + ug → hug
规则 3：   u hugs         hug + s → hugs
最终 token：[u, hugs]
最终 ID：   [5, 9]
```

每一步都检查当前所有相邻对，选择 rank 最小的可用规则，再检查合并后的序列。规则 2 的 `b + ug` 没有出现，所以直接跳过；也不会因为 `uhugs` 看起来像一个完整词而给它临时添加 ID。

### 4.2 编码一整句并保留空格

对于 `"hug bugs"`，先按本文约定预分词：

```text
["hug", " ", "bugs"]
```

然后分别执行 BPE，并连接各片段的结果：

| 片段 | 按规则合并后的 token | ID |
| --- | --- | --- |
| `hug` | `[hug]` | `[7]` |
| 一个空格 | `[" "]` | `[0]` |
| `bugs` | `[bug, s]` | `[8, 4]` |

因此：

```text
encode("hug bugs") = [7, 0, 8, 4]
```

这里词表里即使保留了字符 `h、u、g`，编码器也会根据规则将它们合并为 `hug`。切分依据是规则优先级，不能任意从词表里挑一种拼接方式。

### 4.3 BPE 不是词表最长匹配

可以用另一个小例子看出差异。语料 `cd × 10、abc × 5、bc × 1` 会依次学到 `c+d→cd`、`b+c→bc`、`a+bc→abc`，其中 `cd` 的 rank 最小。

编码 `abcd` 时，BPE 会先合并 `c + d`，得到 `[a, b, cd]`，后续没有可用规则。若从左到右做词表最长匹配，却会选择 `[abc, d]`。两种结果都能拼回同一字符串，但**只有前者遵循这套 BPE 规则**。

## 5. Decode：从 ID 还原文本

### 5.1 查反向词表，再按顺序拼接

建立 `ID → token 内容` 的反向词表，对 `[7, 0, 8, 4]` 逐项查询：

```text
7 → "hug"
0 → " "
8 → "bug"
4 → "s"
```

按原顺序拼接即可：

```text
"hug" + " " + "bug" + "s" = "hug bugs"
```

**解码通常不需要把训练时的合并步骤倒放。** 因为反向词表已经保存了 `hug`、`bug` 等 token 对应的完整字符片段。若某种存储格式只保存递归组成关系，也可以先展开关系；常用 tokenizer 会直接提供 ID 对应的内容。

### 5.2 能恢复文本，不代表能恢复同一套切分

在本文保留字符与空格的条件下，对于支持的文本：

```text
decode(encode(text)) == text
```

但反过来不保证成立。例如，手工构造的 ID `[3, 5, 2]` 对应字符 `[h, u, g]`，解码结果也是 `"hug"`；再次编码却会得到 `[7]`。

```text
decode([3, 5, 2]) = "hug"
encode("hug") = [7]
```

多个不同 ID 序列可能表示同一文本，编码器按固定规则选择其中一种。解码恢复的是内容，不会保留原来的 token 分界，也不会保留“这串可见文字当时是普通 token 还是特殊 token”的完整来源信息。

如果编码前做了不可逆归一化、丢弃空白、未知字符替换或截断，解码就不能保证恢复最初原文。使用特殊 token 时，还应确认解码是否设置了 `skip_special_tokens=True`、空白清理等选项。

## 6. 一个可以运行的完整教学实现

下面代码只依赖 Python 3 标准库，不需要 GPU。将它保存为 `bpe_demo.py`，运行 `python bpe_demo.py` 即可。训练语料写在内存里，不需要下载数据或模型；代码输出的词表、合并记录和 ID 与上面的手算一致。

这是字符级教学实现：基础词表只覆盖本例字符和普通空格，不处理特殊 token。它每轮遍历语料或当前片段，适合小例子；大规模语料应使用经过优化的 tokenizer 训练工具。

```python
import re
from collections import Counter


def pretokenize(text):
    # 保留空白及其数量，BPE 不跨片段合并。
    return re.findall(r"\S+|\s+", text)


def merge_pair(symbols, pair):
    # 从左到右合并所有不重叠的匹配位置。
    result = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and tuple(symbols[index:index + 2]) == pair:
            result.append(pair[0] + pair[1])
            index += 2
        else:
            result.append(symbols[index])
            index += 1
    return result


def train_bpe(piece_frequencies, num_merges):
    # 输入已完成预分词，并将相同片段按出现次数汇总。
    sequences = {piece: list(piece) for piece in piece_frequencies}
    alphabet = sorted(set("".join(piece_frequencies)) | {" "})
    vocab = {symbol: index for index, symbol in enumerate(alphabet)}
    merges = []
    history = []

    for _ in range(num_merges):
        pair_counts = Counter()
        for piece, symbols in sequences.items():
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += piece_frequencies[piece]
        if not pair_counts:
            break

        # 最高频优先；并列时按 pair 的字典序选择。
        pair = min(pair_counts, key=lambda item: (-pair_counts[item], item))
        merged_token = pair[0] + pair[1]
        if merged_token not in vocab:
            vocab[merged_token] = len(vocab)
        merges.append(pair)
        history.append((pair, pair_counts[pair]))
        sequences = {
            piece: merge_pair(symbols, pair)
            for piece, symbols in sequences.items()
        }

    return vocab, merges, history


def encode(text, vocab, merges):
    missing = sorted({symbol for symbol in text if symbol not in vocab})
    if missing:
        raise ValueError(f"教学词表未覆盖这些字符：{missing}")
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    token_ids = []

    for piece in pretokenize(text):
        symbols = list(piece)
        while True:
            candidates = [
                (ranks[pair], pair)
                for pair in zip(symbols, symbols[1:])
                if pair in ranks
            ]
            if not candidates:
                break
            _, pair = min(candidates)
            symbols = merge_pair(symbols, pair)
        token_ids.extend(vocab[symbol] for symbol in symbols)

    return token_ids


def decode(token_ids, vocab):
    inverse_vocab = {token_id: token for token, token_id in vocab.items()}
    return "".join(inverse_vocab[token_id] for token_id in token_ids)


piece_frequencies = {"hug": 10, "hugs": 5, "bug": 6, "bugs": 2}
vocab, merges, history = train_bpe(piece_frequencies, num_merges=4)
print("词表：", vocab)
print("合并记录（pair，频次）：", history)

for text in ("hug bugs", "uhugs", "hug  bugs", ""):
    token_ids = encode(text, vocab, merges)
    restored = decode(token_ids, vocab)
    assert restored == text
    print(repr(text), "→", token_ids, "→", repr(restored))

assert encode("hug bugs", vocab, merges) == [7, 0, 8, 4]
assert encode("uhugs", vocab, merges) == [5, 9]
```

关键输出为：

```text
合并记录（pair，频次）： [(('u', 'g'), 23), (('h', 'ug'), 15), (('b', 'ug'), 8), (('hug', 's'), 5)]
'hug bugs' → [7, 0, 8, 4] → 'hug bugs'
'uhugs' → [5, 9] → 'uhugs'
'hug  bugs' → [7, 0, 0, 8, 4] → 'hug  bugs'
'' → [] → ''
```

注意连续两个空格得到两个 ID `0`，解码后数量仍然保留。初始字符、中间 token 和完整词都留在词表中，只是编码时会按合并规则选择最终切分。

## 7. 对应到 GPT-2 与 `r50k_base`

### 7.1 训练和编码的基础单位改为字节

GPT-2 的普通词表由 256 个基础字节 token 和 50,000 个合并得到的 token 组成；再加一个特殊 token `<|endoftext|>`，总计 50,257 个 ID。训练仍是逐轮统计和合并相邻符号，编码仍是按固定优先级合并，但符号代表的是字节串。

GPT-2 为便于保存词表，将字节可逆映射为内部 Unicode 字符。例如空格字节显示为 `Ġ`，文本 `" the"` 的流程为：

```text
原文：         " the"
UTF-8 字节：  [0x20, 0x74, 0x68, 0x65]
内部基础符号： Ġ t h e
合并：         Ġ t h e → Ġt h e → Ġt he → Ġthe
token ID：     [262]
```

`262` 是 GPT-2 词表里 `Ġthe` 的 ID，不是本文教学词表的编号。tiktoken 的 `r50k_base` 直接使用字节串与 rank/ID 的表来表达同一套普通编码，不需要把字节显示为 `Ġ` 才能工作。

### 7.2 解码应先连接字节，再解码为文本

字节级 BPE 中，一个 token 不一定对应完整的 Unicode 字符。某个中文字符的 UTF-8 字节可能分散在多个 token 中，因此正确的流程为：

```text
token ID 序列 → 每个 token 对应的字节串
             → 将字节串全部连接
             → 按 UTF-8 解码为文本
```

不能假定“对每个 token 单独 UTF-8 解码，再拼接字符串”总能恢复原文。单独解码不完整的字节串可能报错，或者按错误处理设置生成 `�` 替换字符；完整编码结果拼接后的字节才可以恢复原文。

下面用 `r50k_base` 查看字节层的实际内容。需要安装 `tiktoken`；首次加载编码可能需要联网下载词表，也可以使用已有缓存。

```python
import tiktoken

encoding = tiktoken.get_encoding("r50k_base")
text = " the"
token_ids = encoding.encode_ordinary(text)
token_bytes = [encoding.decode_single_token_bytes(token_id) for token_id in token_ids]
restored = b"".join(token_bytes).decode("utf-8")

assert token_ids == [262]
assert restored == encoding.decode(token_ids) == text
print(token_ids)   # [262]
print(token_bytes) # [b' the']
print(repr(restored))  # ' the'
```

### 7.3 文档结束 token 由数据格式决定

预训练数据通常在一条文档的正文编码完成后，显式追加结束 token。例如采用“正文中的特殊 token 字面量按普通文字处理”的策略时：

```python
text = "这是文档正文。"
token_ids = encoding.encode_ordinary(text)
token_ids.append(encoding.eot_token)  # 50256
```

这个 `50256` 是数据处理流程添加的结构信号，不是从文档中的字符频次训练出来的合并规则。解码时如果保留它，通常会得到末尾带 `<|endoftext|>` 的文本；这时编码输入和解码输出相差一个显式添加的边界标记，不能据此认为 BPE 丢失了信息。

追加 EOT 不会自动阻断文档之间的注意力。是否隔离文档、如何设置训练样本边界与注意力掩码，是训练数据组织的另一层配置。

## 8. 实际使用前的检查

| 检查项 | 应确认的内容 |
| --- | --- |
| 训练语料 | 是否代表后续任务中的语言、代码、数字与文本来源；采样比例会影响高频 pair |
| 文本处理 | 归一化和预分词在训练、编码时是否一致；是否丢失空白或大小写 |
| 保存产物 | 词表、合并优先级、特殊 token、预分词与解码配置是否齐全 |
| 编码行为 | 是否使用固定规则；是否额外添加 BOS/EOS；是否开启截断 |
| 解码行为 | 是否跳过特殊 token、清理空白；字节是否先连接再解码 |
| 往返检查 | 在中文、空格、换行、emoji、代码与特殊 token 字面量上检查内容及 ID |
| 模型兼容 | ID 的含义是否与模型嵌入矩阵配套；同样大小的词表不代表相同编码 |

字符级小例子帮助理解算法；真实模型的编码结果还受基础符号、预分词和特殊 token 配置影响。调试时应同时查看 token ID 和对应文本或字节，明确差异发生在哪一步。

## 参考资料

1. Sennrich, Haddow, Birch. [Neural Machine Translation of Rare Words with Subword Units](https://aclanthology.org/P16-1162/). ACL, 2016.
2. Hugging Face. [Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/en/chapter6/5). LLM Course.
3. OpenAI. [GPT-2 编码器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py)，包含字节映射、预分词、BPE 与解码实现。
4. OpenAI. [tiktoken 编码配置](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)与 [encode / decode API 实现](https://github.com/openai/tiktoken/blob/main/tiktoken/core.py)。
