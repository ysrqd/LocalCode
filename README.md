# LocalCode

Local AI coding assistant with web GUI, inspired by Claude Code.

## 核心亮点：每个 Agent 独立 API

本地化 AI 编程工具最大的痛点是什么？**单一模型不够用。** 大模型擅长推理但贵，小模型便宜但笨，视觉模型更是另一套体系。

LocalCode 把三种 Agent **拆开，各自配不同的 API**：

| Agent | 用途 | 推荐配置 |
|-------|------|---------|
| **Main** | 代码生成、工具调用 | Claude / DeepSeek |
| **Intent** | 意图分析、记忆提取 | 便宜的 mini 模型（可选） |
| **Vision** | 图片识别 | Qwen-VL / GPT-4V |

比如你可以：**主力用 DeepSeek，图片识别用通义千问 VL**——DeepSeek 本身不支持视觉，但 LocalCode 的 Vision Subagent 自动把图片发给千问，结果返回给 DeepSeek 继续干活。一套流程，三个模型无缝协作。

## Features

- **Multi-agent API routing**: Main / Intent / Vision 各自独立配置 provider、model、API key
- **Web GUI**: 暗色主题桌面应用，pywebview 构建
- **3-Gate Permission**: Deny list → 规则匹配 → 中文人工确认弹窗（键盘 ↑↓ 选择，Shift+Enter 确认）
- **Memory**: 自动从对话中提取偏好/项目事实，跨会话持久化（Markdown）
- **Skills**: 内置 code-review / explain-code / refactor / write-tests / debug-error
- **MCP**: Model Context Protocol 服务支持
- **Async Settings**: 左侧分类 + 右侧详情，非阻塞加载/保存

## Quick Start

下载 `localcode.exe`，双击启动。首次运行打开 **Settings** 配置 API key。

## Build from Source

```bash
pip install pywebview anthropic openai pyinstaller
python -m PyInstaller --noconsole --onefile --name localcode main_web.py
```

## Project Structure

```
agent.py         # Main agent loop, subagents, memory injection
api.py           # Multi-provider API client (Anthropic / OpenAI)
config.py        # Config persistence (~/.localcode/config.json)
gui_web.py       # Web GUI (pywebview + embedded HTML/CSS/JS)
logger.py        # Structured logging
mcp_client.py    # MCP server protocol
memory.py        # Auto-extract, consolidate, select memories
permission.py    # 3-gate safety pipeline
skills.py        # Skill library (code-review, etc.)
tools.py         # Tool definitions + registry (@tool decorator)
```
