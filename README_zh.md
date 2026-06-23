# 多工具协作的通用 AI Agent

项目简介
--
本项目是基于开源OpenManus的一个多智能体，用于自动操作浏览器完成各种任务，基于 LLM + 多工具协作的设计，支持自主规划、工具调用、多 Agent 协作与沙箱执行。项目目标是让大语言模型像“数字员工”一样，通过工具链完成真实任务（浏览器自动化、代码执行、搜索、文件编辑、可视化等）。

主要特性
--
- 多种入口：CLI（`main.py`）、Web UI（`app.py`）、多 Agent 流程（`run_flow.py`）、MCP 协议
- 丰富工具集：浏览器自动化、Python/Bash 执行、网络爬取、搜索、文件编辑、可视化等
- 支持沙箱执行（Docker）与远程 Daytona 沙箱集成
- 基于 ReAct/ToolCall 模型设计，支持 LLM 的 function-calling 模式

目录概览（简要）
--
- `main.py`：CLI 单 Agent 启动入口
- `app.py`：FastAPI Web UI（含 SSE 实时推送）
- `run_flow.py`：多 Agent 协作入口（PlanningFlow）
- `app/`：核心代码（`agent/`, `tool/`, `flow/`, `sandbox/`, `mcp/`, `prompt/` 等）
- `config/`：配置文件（TOML）
- `tests/`：单元测试

快速开始（本地开发）
--
先决条件：

- Python 3.12+
- Git
- (可选) Docker（用于沙箱）

1. 克隆仓库并进入项目目录：

```bash
git clone <your-repo-url>
cd Multi-agent\ for\ browser\ operation\ based\ on\ Openmanus
```

2. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python -m playwright install    # 如果使用浏览器工具
```

运行示例
--

1) CLI 模式（单 Agent）

```bash
python main.py --prompt "帮我总结一下今天的新闻"
# 或直接运行后根据提示输入问题
python main.py
```

2) Web UI（实时查看 Agent 思考/执行过程）

```bash
python app.py
# 启动后会自动打开浏览器，或访问 http://localhost:5172
```

3) 多 Agent 协作（实验性）

```bash
python run_flow.py
```

配置
--

主要配置文件位于 `config/config.toml`，其中包含 LLM、MCP、服务器端口、浏览器等设置。生产环境请根据需要调整 API 密钥与代理配置。

开发与测试
--

- 运行测试：

```bash
pytest -q
```

- 代码风格与类型检查：请根据团队规则添加工具（black/ruff/mypy）

打包与安装
--

项目包含 `setup.py`，可通过 `pip install -e .` 在本地以可编辑模式安装：

```bash
pip install -e .
```

将项目推送到 GitHub（示例）
--

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:USERNAME/REPO.git
git push -u origin main
```
