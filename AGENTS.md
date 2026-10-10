# Repository Guidelines

## 项目结构与模块组织

本仓库是一个持续建设中的 LLM 学习项目，范围包括预训练、中训练、后训练及相关专题。目前内容主要集中在后训练。顶层目录如下：

- `model_architecture/`：模型架构与关键组件专题，目前仅有目录占位。
- `tokenization/`：分词与 tokenizer 专题，已有主流方法、发展历程与工程核查教程。
- `pretrain/`：预训练专题，已有 Pile 原始数据来源说明。
- `midtrain/`：中训练与继续预训练专题，目前仅有目录占位。
- `sft/`：监督微调基础理论与实验规划。
- `dpo/`：Direct Preference Optimization 基础理论。
- `opd/`：在线策略蒸馏相关材料，目前只有占位入口。
- `rl/`：强化学习方法，目前包含 PPO 和 GRPO 基础理论。
- `evaluation/`：模型评测专题，目前仅有目录占位。
- `inference/`：模型推理与部署专题，目前仅有目录占位。
- `utils/`：通用辅助脚本，目前包含 `utils/dataset_utils/`。

预留目录目前仅包含 `.gitkeep`；开始撰写专题内容时再添加 `readme.md`，说明主题范围。其他新方向在有实际材料时再建目录。现有后训练目录不因项目范围扩大而整体迁移，以免打断已有链接。

目录和文件名优先使用小写。主题入口文档优先使用 `readme.md`，除非后续全仓库统一改为 `README.md`。

## 构建、测试与开发命令

当前没有全仓库级别的构建系统。小型工具脚本可直接用 Python 运行：

```bash
python utils/dataset_utils/download_data_from_hf.py
python utils/dataset_utils/load_dataset_from_disk.py
```

新增训练命令前，应在对应主题的 `readme.md` 中说明依赖、GPU/内存需求、输入路径和输出路径。

## 代码风格与命名规范

Python 代码应保持简单、清晰，适合教程阅读。使用 4 空格缩进、描述性的 `snake_case` 命名；脚本超过短示例规模后，应拆分为小函数。提交示例代码时避免写死个人绝对路径，优先使用变量、环境变量，或明确标注的占位路径，例如 `/path/to/dataset`。

后续文档、代码注释和教程说明默认使用中文。专业术语可保留常见英文写法，例如 SFT、DPO、RLHF、LoRA、Hugging Face。

不同章节中的公式符号必须兼容、一致。新增或改写公式时，应检查已有章节的变量命名、上下标和目标函数记号，避免同一概念使用多个符号，或同一符号表示不同含义。

Markdown 公式应同时兼容 GitHub 与 VS Code/Cursor 的标准 Markdown Preview。行内公式使用 `$...$` 时，公式前后的 `$` 都不要紧贴中文标点，应与正文保留一个空格；不要在 `$` 与公式内容之间额外加空格。包含多层上下标、分式或较长表达式的公式优先使用独立公式块，避免 GitHub 行内解析不稳定。行间公式的 `$$` 必须单独成行，并在公式块前后保留空行。LaTeX 公式中不要直接写 `<`，因为 GitHub 可能先将其解析为 HTML；小于号统一写成 `\lt`。GitHub 禁止 `\operatorname` 宏，arg min/max 应写成 `\underset{\theta}{\arg\min}` 或 `\underset{\theta}{\arg\max}`。修改公式后，应检查是否仍有紧贴中文标点的行内公式、数学环境中的裸 `<`，或 GitHub 不允许的宏。

Cursor 的编辑区内置了带有 “Preview / Markdown” 切换按钮的 WYSIWYG Markdown 预览，但该预览器与 VS Code 标准 Markdown Preview 不是同一个组件，可能直接显示 LaTeX 源码，且不受 `markdown.math.enabled` 控制。即使 `.vscode/settings.json` 已将 `*.md` 默认关联到 `vscode.markdown.preview.editor`，从源码点击 Cursor 右上角的 “Preview” 仍会切回不支持公式的 WYSIWYG 预览；文件关联只能保证重新打开文件时进入标准预览，不能改变这个按钮的行为。

在 Cursor 中编辑含公式的 Markdown 时，不要使用右上角的 “Preview / Markdown” 按钮往返切换。源码编辑完成后，应使用 `Cmd+Shift+V` 执行 “Markdown: Open Preview”，或使用 `Cmd+K` 后按 `V` 执行 “Markdown: Open Preview to the Side”；这两个命令会调用支持 `markdown.math.enabled` 的标准 Markdown Preview。若需要从标准预览返回源码，可执行 “Markdown: Show Source” 或 “Reopen Editor With... → Text Editor”。若必须彻底禁用 Cursor WYSIWYG，只能使用 Cursor 的 classic 窗口模式启动项目；不要依赖未公开的 feature gate 或修改 Cursor 安装文件。

## 测试规范

当前尚未配置正式测试框架。向 `utils/` 添加可复用代码时，应提供可本地运行的轻量检查或示例。若后续引入测试套件，请放在 `tests/` 目录下，测试文件命名为 `test_*.py`，并记录运行命令，例如：

```bash
pytest
```

教程命令优先提供小型冒烟测试，用于验证依赖导入、数据集加载和输出目录创建。

## 提交与 Pull Request 规范

请使用清晰的约定式提交信息，例如 `docs: expand sft tutorial` 或 `feat: add dataset loader utility`。

Pull Request 应包含简短摘要、变更目录、已运行命令，以及刻意排除在 Git 之外的生成产物说明。如有关联 issue，请一并链接。不要提交模型权重、checkpoint、本地数据集、缓存、日志或私密凭据。

## Agent 专用说明

Codex 或其他 AI agent 在编辑前应先检查仓库结构，并保持改动范围聚焦。优先保证教程可读性，不要为了“聪明”的抽象牺牲清晰度。若同时修改代码和文档，必须确保文档中的命令与实际脚本保持一致。
