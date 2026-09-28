# Markdown 编写规范：兼容 GitHub 与 MarkText

本文记录本仓库目前验证过的 Markdown 兼容规则。目标是在 GitHub 和 MarkText 中获得尽量一致的阅读体验，尤其保证标题、表格、代码和数学公式能够正常显示。

## 1. 基本原则

1. 优先使用 GitHub Flavored Markdown（GFM）的通用语法。
2. 不依赖只被某一个编辑器支持的私有扩展。
3. 数学公式使用 GitHub 和 MarkText 都能识别的 LaTeX 子集。
4. 写完后分别在 MarkText 和 GitHub 中检查；本地正常不代表 GitHub 一定正常。
5. 标题是文档结构的唯一事实来源，不维护容易过期的手写目录。

## 2. 标题与文档结构

使用 ATX 风格标题，即行首的 `#`：

```markdown
# 文档标题

## 一级章节

### 二级小节
```

注意事项：

- `#` 与标题文字之间保留一个空格。
- 标题级别逐级递进，避免从 `##` 直接跳到 `####`。
- 一个文档通常只使用一个一级标题 `#`。
- 不使用 Setext 风格标题，即不要用下一行的 `===` 或 `---` 表示标题。

不推荐：

```markdown
文档标题
========
```

Setext 标题容易与公式中的独立 `=`、分隔线等内容产生歧义。

## 3. 目录

### 3.1 不使用 `[TOC]`

`[TOC]` 不是 CommonMark 或 GFM 标准。当前使用的 MarkText 版本和 GitHub 都不会把它稳定地展开为正文目录，因此本仓库不使用：

```markdown
[TOC]
```

### 3.2 推荐使用自动大纲

- MarkText：使用 `View → Toggle Table of Contents` 打开侧边栏大纲。
- GitHub：使用文件页面右上角的 Outline（大纲）按钮。

两者都会根据标题自动更新，不需要在新增、删除或重命名章节后维护目录。

### 3.3 谨慎使用手写目录

GitHub 和当前 MarkText 对中文标题生成锚点的算法不同。例如标题：

```markdown
## 1. 为什么需要 IC？
```

可能得到不同锚点：

- GitHub：`#1-为什么需要-ic`
- 当前 MarkText：可能删除中文，只留下类似 `#1-ic`

因此，中文标题的手写目录链接无法保证同时在两端跳转。

也不要用原始 HTML 强行添加锚点：

```html
<a id="section-1"></a>
```

当前 MarkText 可能把它直接显示为文本，并且不将其作为可跳转目标。

如果将来必须在正文中嵌入跨平台目录，可以考虑以下方案之一：

1. 将章节标题改成稳定的纯英文标题；
2. 接受只以 GitHub 为目标，由工具生成目录；
3. 使用文档站点生成器统一渲染，而不依赖两个阅读器各自的锚点算法。

## 4. 行内数学公式

行内公式使用一对单美元符号：

```markdown
因子值记为 $x_{i,t}$，未来收益记为 $r_{i,t\rightarrow t+h}$。
```

注意事项：

- 起止 `$` 必须成对出现。
- 不要在 `$` 与公式之间添加多余空格。
- 对复杂变量定义，建议每句话只放一个行内公式，降低解析器产生边界歧义的概率。
- 如果一行中某个公式没有渲染，先检查其中是否包含 GitHub 不允许的宏，而不要只检查 `$` 是否成对。

推荐：

```markdown
其中，$\mathbf{z}_{i,t}$ 表示行业和市值等风险暴露。

回归残差记为 $\varepsilon_{i,t}$，它是中性化后的因子。
```

不推荐把多个复杂定义挤在同一个长句中。

## 5. 行间数学公式

### 5.1 统一使用三行式

行间公式统一采用下面的结构：

```markdown
$$
a_b
$$
```

其中：

- 开始和结束的 `$$` 分别独占一行；
- 公式主体只占一个物理行；
- `$$` 前后各保留一个空行。

推荐：

```markdown
$$
r_{i,t\rightarrow t+h}=\frac{P_{i,t+h}}{P_{i,t}}-1
$$
```

### 5.2 不把等号单独放一行

不要写成：

```markdown
$$
r_{i,t\rightarrow t+h}
=
\frac{P_{i,t+h}}{P_{i,t}}-1
$$
```

部分 Markdown 解析器会在数学渲染之前，将独立的 `=` 误识别成 Setext 标题下划线，导致公式被拆成标题和普通文本。

因此，即使公式很长，也优先把公式主体保持在同一个物理行：

```markdown
$$
r_{i,t\rightarrow t+h}=\frac{P_{i,t+h}}{P_{i,t}}-1
$$
```

### 5.3 多步推导

需要展示多步推导时，可以在同一物理行内使用 `aligned`：

```markdown
$$
\begin{aligned} RankIC &=1-\frac{6\times4}{5\times(5^2-1)} \\ &=1-\frac{24}{120} \\ &=0.8 \end{aligned}
$$
```

这样既能显示多行排版，又不会在 Markdown 源文件中产生独立的 `=` 行。

## 6. 使用 GitHub 支持的数学宏

MarkText 和 GitHub 使用的数学渲染器或安全配置不同。MarkText 能显示的 LaTeX 宏，不一定会被 GitHub 接受。

### 6.1 避免 `\operatorname`

GitHub 可能返回如下错误：

```text
The following macros are not allowed: operatorname
```

不要写：

```latex
\operatorname{Corr}(x,r)
\operatorname{Rank}(x)
\operatorname{Median}(x)
```

改用：

```latex
\mathrm{Corr}(x,r)
\mathrm{Rank}(x)
\mathrm{Median}(x)
```

完整示例：

```markdown
$$
IC_t=\mathrm{Corr}\left(x_{i,t},r_{i,t\rightarrow t+h}\right)
$$
```

### 6.2 行内公式避免 `\boldsymbol`

当前实践中，GitHub 可能无法正常渲染包含 `\boldsymbol` 的行内公式，而 MarkText 可以。

对于拉丁字母向量，不要写：

```latex
\boldsymbol{z}_{i,t}
```

改用：

```latex
\mathbf{z}_{i,t}
```

对于希腊字母向量，`\mathbf` 通常不会产生预期的粗体效果。若确实需要粗体希腊字母，应先在 GitHub 中验证所选宏；否则优先用正文说明“该符号为向量”，不要为了样式牺牲兼容性。

### 6.3 当前已验证的常用宏

本仓库当前已使用并正常工作的常用写法包括：

```latex
\frac{a}{b}
\sqrt{x}
\sum_{i=1}^{N} x_i
\bar{x}
\overline{IC}
\mathrm{Corr}(x,r)
\mathbf{z}
\varepsilon
\alpha
\beta
\rightarrow
\approx
\left( \right)
\begin{aligned} ... \end{aligned}
```

这不是 GitHub 支持宏的完整列表，只代表本仓库目前的实际使用经验。新增不常见宏后必须在 GitHub 上验证。

## 7. 表格

使用标准 GFM 表格，并在表格前后保留空行：

```markdown
| 指标 | 含义 |
|---|---|
| IC | 因子值与未来收益的截面相关性 |
| ICIR | IC 均值与 IC 标准差之比 |
```

注意事项：

- 表头分隔行不可省略。
- 每一行保持相同列数。
- 尽量不要在表格单元格中放复杂块级公式或多行代码。
- 行内公式可以放在表格中，但仍要使用受支持的宏。

## 8. 代码

行内代码使用反引号：

```markdown
数据列名为 `forward_return`。
```

代码块使用三个反引号并注明语言：

````markdown
```python
mean_ic = ic_series.mean()
```
````

注意事项：

- 起止围栏必须成对出现。
- 数学公式不要放进代码块，否则只会显示源码，不会进行数学渲染。
- 展示 Markdown 源码时，外层围栏应比示例中的围栏更长，例如使用四个反引号。

## 9. 列表、空行与分隔线

### 9.1 列表

无序列表使用 `-`，有序列表使用 `1.`、`2.`：

```markdown
- 第一项
- 第二项

1. 第一步
2. 第二步
```

列表前保留空行，避免被拼接到上一段。

### 9.2 分隔线

分隔线使用：

```markdown
---
```

分隔线前后保留空行，避免它被解释成 Setext 标题的一部分。

### 9.3 段落

段落之间使用一个空行，不依赖行尾两个空格制造换行。短句、公式解释和定义可以适当拆段，以提升源文件可维护性和渲染稳定性。

## 10. 常见故障与定位方法

### 10.1 公式显示成标题和普通文本

症状：`$$`、公式第一行和剩余内容被拆开，公式第一部分变成大标题。

常见原因：公式中的 `=` 单独占一行，被识别为 Setext 标题标记。

解决方法：使用三行式公式，并让公式主体保持在一个物理行内。

### 10.2 GitHub 提示某个宏不允许

症状：GitHub 显示红色错误框，例如：

```text
The following macros are not allowed: operatorname
```

原因：GitHub 的数学渲染器使用宏白名单或安全限制。

解决方法：用更基础的宏替换。例如将 `\operatorname{Corr}` 改成 `\mathrm{Corr}`。

### 10.3 MarkText 正常，GitHub 显示公式源码

原因通常不是 `$` 本身，而是公式中存在 GitHub 不支持的宏。

排查顺序：

1. 检查 `$` 或 `$$` 是否成对；
2. 检查块公式是否符合三行式；
3. 检查公式主体是否被拆成多行，尤其是否存在独立的 `=`；
4. 检查 `\operatorname`、`\boldsymbol` 等兼容性较差的宏；
5. 将复杂公式逐步简化，定位具体失败宏。

### 10.4 目录链接在 GitHub 可用，在 MarkText 不可用

原因：两者对中文标题生成的锚点不同。

解决方法：不在正文中维护手写中文目录，改用各自的自动大纲。

### 10.5 原始 HTML 直接显示出来

原因：MarkText 对某些原始 HTML 元素的处理与 GitHub 不同。

解决方法：避免依赖 `<a id="...">` 等 HTML 扩展，优先使用纯 Markdown。

## 11. 发布前检查清单

提交 Markdown 文档前，至少检查以下内容：

- [ ] 标题使用 `#` 风格，并且层级合理；
- [ ] 没有使用 `[TOC]`；
- [ ] 没有使用原始 HTML 锚点；
- [ ] 行内公式的 `$` 成对出现；
- [ ] 行间公式使用三行式；
- [ ] 每个行间公式的主体只占一个物理行；
- [ ] 公式中没有独立一行的 `=`；
- [ ] 没有使用 GitHub 禁止的 `\operatorname`；
- [ ] 行内向量优先使用 `\mathbf`，谨慎使用 `\boldsymbol`；
- [ ] 代码围栏成对，并注明语言；
- [ ] 表格每行列数一致；
- [ ] 分隔线前后存在空行；
- [ ] 在 MarkText 中检查公式、表格和大纲；
- [ ] 在 GitHub 中检查公式、表格和大纲；
- [ ] 执行 `git diff --check`，确保没有多余行尾空格等问题。

## 12. 推荐模板

新建教程文档时，可以从下面的最小模板开始：

````markdown
# 文档标题

## 1. 本章目标

用一段话说明本章要解决的问题。

## 2. 核心概念

因子值记为 $x_{i,t}$。

未来收益定义为：

$$
r_{i,t\rightarrow t+h}=\frac{P_{i,t+h}}{P_{i,t}}-1
$$

## 3. Python 实现

```python
forward_return = future_price / current_price - 1
```

## 4. 小结

- 结论一；
- 结论二。
````

