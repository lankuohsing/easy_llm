# Pile-uncopyrighted 预训练数据集

## 数据集概览

[monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted) 是 [The Pile](https://arxiv.org/abs/2101.00027) 的一个多来源文本语料发布版本，包含网页、百科、科研论文、代码、问答、法律文书和对话等内容。它适合用来学习从原始文本到清洗、分词和下一 token 预测训练的预训练流程。记录没有统一的“指令—回答”结构。实际使用时，应查阅数据集页面及各来源的许可信息。

| 项目 | 内容 |
| --- | --- |
| 数据集标识 | `monology/pile-uncopyrighted` |
| 原始格式 | zstd 压缩的 JSONL，文件后缀为 `.jsonl.zst` |
| 训练文件 | `train/00.jsonl.zst` 至 `train/29.jsonl.zst`，共 30 个分片；可在[文件列表](https://hf-mirror.com/api/datasets/monology/pile-uncopyrighted/tree/main/train?recursive=true)核对 |
| 验证文件 | `val.jsonl.zst`，与训练分片分别存放 |
| 测试文件 | `test.jsonl.zst`，与训练、验证文件分别存放 |
| 原始数据文件合计 | 32 个 `.jsonl.zst` 文件：30 个训练分片、1 个验证文件、1 个测试文件 |

数据集仓库另有 `README.md` 和 `.gitattributes` 两个说明或配置文件，不属于上述 32 个原始数据文件。数据集页面显示的 **Total file size 约 335 GB** 可作为压缩文件下载量的参考。后续解压、清洗或生成分词数据，需要另外预留磁盘空间。记录数、压缩字节数和 token 数是不同的统计口径，不能相互直接换算。

## 原始记录结构

解压 `.jsonl.zst` 后得到 JSONL：每行是一条独立的 JSON 记录，主要结构如下：

```json
{"text": "文本内容……", "meta": {"pile_set_name": "ArXiv"}}
```

`text` 是文本内容，`meta.pile_set_name` 是子语料的来源标签。标签说明来源，但不保证每条记录的语言、格式或质量一致。单条记录的长度差异很大，论文、图书和聊天日志可能远长于摘要或问答。要直接阅读原始内容，需要先解压或使用流式解压工具。

下表的“验证集记录数”来自完整 `val.jsonl.zst` 的本地统计，共 179,996 条；它不代表训练集中的来源比例。记录数也不能用于估算各来源的 token 占比。

## 验证文件中的 17 个来源

| `pile_set_name` | 来源与典型内容 | 人工查看时留意 | 验证集记录数 |
| --- | --- | --- | ---: |
| Pile-CC | 从 Common Crawl 网页语料筛选出的通用网页文本；主题广泛。 | 网页正文可能混有导航、广告或抓取残留。 | 52,792 |
| StackExchange | Stack Exchange 各站点的问答，涉及编程、科学等主题。 | 问题、答案、代码片段和格式标记可能出现在同一条文本中。 | 29,950 |
| PubMed Abstracts | PubMed 收录的生物医学论文摘要。 | 通常是摘要文本，不能按完整论文正文理解。 | 29,871 |
| Github | GitHub 仓库中的源代码及相关文本。 | 语法、注释、许可证头和自然语言混合出现。 | 18,337 |
| Wikipedia (en) | 英文维基百科条目。 | 可见标题、段落及百科式叙述，也可能有排版残留。 | 17,478 |
| USPTO Backgrounds | 美国专利文本中的背景技术部分。 | 常见专利术语和固定写法，内容侧重技术背景。 | 11,387 |
| PubMed Central | PubMed Central 收录的生物医学论文正文。 | 篇幅通常比摘要长，可能保留章节标题、引文及排版标记。 | 5,977 |
| FreeLaw | 公开的法律文书，样本中可见法院判例。 | 引用格式、案号和法律术语较多。 | 5,094 |
| ArXiv | arXiv 科研论文的正文文本。 | 可能包含公式、引用和从原格式转换来的标记。 | 2,434 |
| DM Mathematics | DeepMind Mathematics 数据中的数学题与答案文本。 | 更接近结构化习题，不宜当作普通散文。 | 2,007 |
| NIH ExPorter | 美国 NIH ExPORTER 的研究项目描述等文本。 | 属于科研资助项目信息，与 PubMed 论文摘要不同。 | 1,825 |
| HackerNews | Hacker News 的帖子与讨论文本。 | 链接、标题、评论及网页引用可能交织。 | 1,619 |
| Enron Emails | Enron 邮件语料中的工作邮件。 | 邮件头、转发链和引用内容可能重复。 | 947 |
| EuroParl | 欧洲议会会议发言记录。 | 包含多种欧洲语言；本地样本中能看到波兰语。 | 133 |
| PhilPapers | 哲学领域的论文和学术文章。 | 篇幅、引文和论文格式与网页文本差异较大。 | 64 |
| Gutenberg (PG-19) | Project Gutenberg 图书构成的 PG-19 长文本语料。 | 一条记录可能很长，也可能含电子书制作说明。 | 60 |
| Ubuntu IRC | Ubuntu 社区 IRC 频道的聊天记录。 | 带有频道、时间或昵称等对话格式；本地样本中有意大利语。 | 21 |

这些来源的文本结构差异很大。处理时应留意网页残留、代码格式、论文公式、邮件转发链和聊天标记；也不能仅凭数据集名称假设每条记录都是英文。

## 下载方式

在 Linux 上使用仓库中的 [下载脚本](download_pile_uncopyrighted.sh)。依赖 Bash、`curl`、`sha256sum` 和常见的系统命令，不需要 Python。把下面的目标路径改成服务器上的数据目录：

```bash
bash pretrain/download_pile_uncopyrighted.sh /path/to/pile
```

脚本默认通过 `hf-mirror.com` 下载 30 个训练分片、验证文件和测试文件。镜像中的验证、测试文件位于数据集根目录（`val.jsonl.zst` 和 `test.jsonl.zst`）；下载后分别保存在目标目录的 `val/`、`test/` 下，训练分片保存在 `train/` 下。镜像可能继续跳转到实际存储文件的域名，因此这些域名也需要可连接。

下载中断时，脚本保留 `.part` 文件；重试或重新运行同一命令时从已下载的字节续传。它会核对远端文件大小和 SHA-256，只有校验通过才改成 `.jsonl.zst` 正式文件名。重新运行时也会校验已有文件，因此扫描大文件需要一些时间。HTTP 403/404、文件大小或校验值异常会停止并保留现场，避免误把错误页面或损坏文件当作数据。每个文件默认最多尝试 10 次；如需调整，可在命令前设置 `PILE_MAX_ATTEMPTS=20`。

完整压缩数据约 335 GB。运行前检查目标磁盘空间，并为后续解压、清洗和分词预留额外空间。

### 连通性排查

下载前可只请求文件的响应头：

```bash
curl -IL --connect-timeout 10 --max-time 30 \
  'https://hf-mirror.com/datasets/monology/pile-uncopyrighted/resolve/main/train/00.jsonl.zst'
```

`-I` 只请求响应头，`-L` 跟随跳转。镜像返回 `302`、最终存储地址返回 `200` 是正常的；如果最终存储域名无法连接，实际下载仍会失败。响应头成功也不能保证大文件传输稳定。复制命令时应使用代码块中的普通 URL，不要把 `[https://...](https://...)` 形式的 Markdown 链接文本粘进终端。

## 后续处理

下载得到的仍是压缩的原始 JSONL。训练前通常需要流式解压、检查或清洗文本，再进行分词，并把 token 写入训练程序可读取的格式。处理时应保留训练集与验证集的划分，并根据实际 token 数规划存储与训练规模。
