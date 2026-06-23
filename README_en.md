# Multi-Tool Collaborative General AI Agent

Project Overview
--
This project is a multi-agent system based on the open-source OpenManus, designed to automate browser operations and complete various tasks. It follows an LLM + multi-tool collaboration design, supporting autonomous planning, tool invocation, multi-agent cooperation, and sandboxed execution. The goal is to make large language models behave like "digital workers" by completing real tasks through a toolchain (browser automation, code execution, web search, file editing, visualization, etc.).

Key Features
--
- Multiple entry points: CLI (`main.py`), Web UI (`app.py`), multi-agent flow (`run_flow.py`), and MCP protocol
- Rich toolset: browser automation, Python/Bash execution, web crawling, search, file editing, visualization, and more
- Sandbox execution support (Docker) and remote Daytona sandbox integration
- Built on ReAct/ToolCall design and supports LLM function-calling

Directory Overview (brief)
--
- `main.py`: CLI entry point for single-agent mode
- `app.py`: FastAPI Web UI (with SSE real-time events)
- `run_flow.py`: Multi-agent collaboration entry (PlanningFlow)
- `app/`: Core code (agent/, tool/, flow/, sandbox/, mcp/, prompt/, etc.)
- `config/`: Configuration files (TOML)
- `tests/`: Unit tests

Quick Start (local development)
--
Prerequisites:

- Python 3.12+
- Git
- (Optional) Docker (for sandbox)

1. Clone the repository and enter the project folder:

```bash
git clone <your-repo-url>
cd Multi-agent\ for\ browser\ operation\ based\ on\ Openmanus
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
python -m playwright install    # If you plan to use browser tools
```

Run Examples
--

1) CLI mode (single agent)

```bash
python main.py --prompt "Summarize today's news for me"
# Or run and follow the prompt to enter your request
python main.py
```

2) Web UI (watch agent thinking/executing in real time)

```bash
python app.py
# The server will open a browser automatically, or visit http://localhost:5172
```

3) Multi-agent collaboration (experimental)

```bash
python run_flow.py
```

Configuration
--

The main configuration file is `config/config.toml`, which includes settings for LLM, MCP, server port, browser, etc. Adjust API keys and proxy settings for production use.

Development & Testing
--

- Run tests:

```bash
pytest -q
```

- Code style and type checking: add tools such as `black`, `ruff`, and `mypy` according to your team's convention.

Packaging & Installation
--

The project contains `setup.py`. You can install it in editable mode locally:

```bash
pip install -e .
```

Push to GitHub (example)
--

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:USERNAME/REPO.git
git push -u origin main
```
