"""
app.py —— Web UI 入口

这个文件实现了一个 Web 界面，让你可以通过浏览器与 Manus Agent 交互。
启动方式：
    python app.py
    # 会自动打开浏览器访问 http://localhost:5172

技术栈：
  - FastAPI：现代 Python Web 框架（类似 Flask 但支持异步）
  - SSE（Server-Sent Events）：服务端推送事件，用于实时显示 Agent 的思考/执行过程
  - Jinja2：模板引擎，用于渲染 HTML 页面
  - Uvicorn：ASGI 服务器，运行 FastAPI 应用

整体架构：
  前端页面 --HTTP POST--> /tasks（创建任务）
                        |
                        v
                   TaskManager（管理任务队列）
                        |
                        v
                   Manus Agent（执行 think→act 循环）
                        |
  前端页面 <--SSE---- /tasks/{id}/events（实时推送思考/执行过程）
"""

import asyncio         # 异步 I/O 框架
import os              # 操作系统接口
import threading       # 多线程（用于启动浏览器）
import tomllib         # TOML 配置文件解析
import uuid            # 生成唯一 ID
import webbrowser      # 自动打开浏览器
import json            # JSON 处理
from datetime import datetime  # 时间处理
from functools import partial  # 函数部分应用
from json import dumps         # JSON 序列化
from pathlib import Path       # 文件路径处理

# FastAPI 相关导入
from fastapi import Body, FastAPI, HTTPException, Request  # Web 框架核心
from fastapi.middleware.cors import CORSMiddleware          # 跨域支持
from fastapi.responses import (                              # 响应类型
    FileResponse,       # 文件下载响应
    HTMLResponse,       # HTML 响应
    JSONResponse,       # JSON 响应
    StreamingResponse,  # 流式响应（用于 SSE）
)
from fastapi.staticfiles import StaticFiles   # 静态文件服务
from fastapi.templating import Jinja2Templates  # 模板引擎
from pydantic import BaseModel                 # 数据模型基类

# 创建 FastAPI 应用实例
app = FastAPI()

# 挂载静态文件目录（CSS、JS、图片等），访问 /static/xxx 会映射到 static/xxx
app.mount("/static", StaticFiles(directory="static"), name="static")
# 初始化 Jinja2 模板引擎，HTML 模板存放在 templates/ 目录
templates = Jinja2Templates(directory="templates")

# 配置 CORS（跨域资源共享）
# 开发环境下，前端可能运行在不同端口，需要允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许所有来源（生产环境应限制）
    allow_credentials=True,    # 允许携带 Cookie
    allow_methods=["*"],       # 允许所有 HTTP 方法
    allow_headers=["*"],       # 允许所有请求头
)


# ========== 任务数据模型 ==========
class Task(BaseModel):
    """
    任务数据模型，表示一个用户提交的任务。
    每个任务包含：
      - id: 唯一标识符（UUID）
      - prompt: 用户输入的问题/需求
      - created_at: 创建时间
      - status: 状态（pending/running/completed/failed）
      - steps: 执行步骤列表，每个步骤包含 type、step、result
    """
    id: str
    prompt: str
    created_at: datetime
    status: str
    steps: list = []

    def model_dump(self, *args, **kwargs):
        """重写序列化方法，将 datetime 转为 ISO 格式字符串"""
        data = super().model_dump(*args, **kwargs)
        data["created_at"] = self.created_at.isoformat()
        return data


# ========== 任务管理器 ==========
class TaskManager:
    """
    任务管理器 —— 管理所有任务的创建、执行和状态更新。

    核心数据结构：
      - tasks: {任务ID: Task对象} 字典，存储所有任务
      - queues: {任务ID: asyncio.Queue} 字典，每个任务有自己的消息队列
               用于 SSE 推送（Agent 每完成一步，就把结果放入队列，前端从队列读取）
    """

    def __init__(self):
        self.tasks = {}    # 存储所有任务
        self.queues = {}   # 每个任务的消息队列（用于 SSE 推送）

    def create_task(self, prompt: str) -> Task:
        """创建一个新任务，并为其分配唯一 ID 和消息队列"""
        task_id = str(uuid.uuid4())  # 生成 UUID 作为任务 ID
        task = Task(
            id=task_id, prompt=prompt, created_at=datetime.now(), status="pending"
        )
        self.tasks[task_id] = task
        self.queues[task_id] = asyncio.Queue()  # 为该任务创建独立的消息队列
        return task

    async def update_task_step(
        self, task_id: str, step: int, result: str, step_type: str = "step"
    ):
        """
        更新任务的某个步骤结果。
        这个方法会被 SSE 推送调用：
          1. 把步骤结果添加到 task.steps 列表
          2. 把步骤结果放入消息队列（前端通过 SSE 读取）
          3. 把当前状态也放入队列（前端可以更新进度条）
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.steps.append({"step": step, "result": result, "type": step_type})
            # 推送步骤结果到队列
            await self.queues[task_id].put(
                {"type": step_type, "step": step, "result": result}
            )
            # 推送当前状态到队列
            await self.queues[task_id].put(
                {"type": "status", "status": task.status, "steps": task.steps}
            )

    async def complete_task(self, task_id: str, result: str):
        """标记任务为已完成"""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = "completed"
            await self.queues[task_id].put(
                {"type": "status", "status": task.status, "steps": task.steps}
            )
            await self.queues[task_id].put({"type": "complete", "result": result})

    async def fail_task(self, task_id: str, error: str):
        """标记任务为失败"""
        if task_id in self.tasks:
            self.tasks[task_id].status = f"failed: {error}"
            await self.queues[task_id].put({"type": "error", "message": error})


# 创建全局任务管理器实例
task_manager = TaskManager()


# ========== API 路由 ==========

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """首页路由，返回 HTML 页面"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/download")
async def download_file(file_path: str):
    """文件下载接口，允许用户下载 Agent 生成的文件"""
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=os.path.basename(file_path))


@app.post("/tasks")
async def create_task(prompt: str = Body(..., embed=True)):
    """
    创建新任务的 API。
    前端发送 POST 请求，包含用户的 prompt，
    后端创建任务并立即启动一个异步任务（asyncio.create_task）来运行 Agent。
    """
    task = task_manager.create_task(prompt)
    # asyncio.create_task() 会在后台启动任务，不阻塞当前请求
    asyncio.create_task(run_task(task.id, prompt))
    return {"task_id": task.id}  # 返回任务 ID，前端用它来订阅 SSE


# 导入 Manus Agent（放在这里而不是文件顶部，避免循环导入）
from app.agent.manus import Manus


async def run_task(task_id: str, prompt: str):
    """
    运行单个任务的核心逻辑。
    这个函数在后台异步执行，流程：
      1. 创建 Manus Agent
      2. 注册日志处理器（捕获 Agent 的思考/执行日志）
      3. 运行 agent.run(prompt)
      4. 通过 TaskManager 将结果推送到前端
    """
    try:
        task_manager.tasks[task_id].status = "running"

        # 创建 Manus Agent（注意：这里没用 create()，而是直接实例化，
        # MCP 服务器会在第一次 think() 时延迟初始化）
        agent = Manus(
            name="Manus",
            description="A versatile agent that can solve various tasks using multiple tools",
        )

        # 以下是各种事件回调函数，当 Agent 执行不同阶段时会调用它们
        # 每个回调都会把信息推送到前端（通过 TaskManager）

        async def on_think(thought):
            """Agent 思考时的回调"""
            await task_manager.update_task_step(task_id, 0, thought, "think")

        async def on_tool_execute(tool, input):
            """Agent 执行工具时的回调"""
            await task_manager.update_task_step(
                task_id, 0, f"Executing tool: {tool}\nInput: {input}", "tool"
            )

        async def on_action(action):
            """Agent 执行动作时的回调"""
            await task_manager.update_task_step(
                task_id, 0, f"Executing action: {action}", "act"
            )

        async def on_run(step, result):
            """每个步骤完成时的回调"""
            await task_manager.update_task_step(task_id, step, result, "run")

        # SSELogHandler: 日志处理器，捕获 Agent 的日志并转换为 SSE 事件
        # 它通过解析日志消息中的关键字来判断事件类型：
        #   - "Manus's thoughts" -> think 事件（思考）
        #   - "selected" -> tool 事件（选择工具）
        #   - "Tool" -> act 事件（执行工具）
        #   - "Oops" -> error 事件（错误）
        #   - "Special tool" -> complete 事件（完成）
        from app.logger import logger

        class SSELogHandler:
            def __init__(self, task_id):
                self.task_id = task_id

            async def __call__(self, message):
                """当日志产生时被调用，解析日志消息并推送到前端"""
                import re

                # 清理日志消息前缀（去掉时间戳等前缀）
                cleaned_message = re.sub(r"^.*? - ", "", message)

                # 根据消息内容判断事件类型
                event_type = "log"
                if "✨ Manus's thoughts:" in cleaned_message:
                    event_type = "think"     # 思考事件
                elif "🛠 Manus selected" in cleaned_message:
                    event_type = "tool"      # 工具选择事件
                elif "🎯 Tool" in cleaned_message:
                    event_type = "act"       # 动作执行事件
                elif "📝 Oops!" in cleaned_message:
                    event_type = "error"     # 错误事件
                elif "🏁 Special tool" in cleaned_message:
                    event_type = "complete"  # 完成事件

                await task_manager.update_task_step(
                    self.task_id, 0, cleaned_message, event_type
                )

        # 注册日志处理器，将 Agent 的日志通过 SSE 推送到前端
        sse_handler = SSELogHandler(task_id)
        hwnd = logger.add(sse_handler)  # 添加到日志系统

        # 运行 Agent（这是核心调用，会执行 think→act 循环）
        result = await agent.run(prompt)

        # 移除日志处理器，推送最终结果
        logger.remove(hwnd)
        await task_manager.update_task_step(task_id, 1, result, "result")
        await asyncio.sleep(3)  # 等待 3 秒，确保前端收到所有事件
        await task_manager.complete_task(task_id, result)
    except Exception as e:
        # 发生异常时标记任务失败
        await task_manager.fail_task(task_id, str(e))


@app.get("/tasks/{task_id}/events")
async def task_events(task_id: str):
    """
    SSE（Server-Sent Events）端点 —— 实时推送任务执行过程。

    前端通过 EventSource 连接这个端点，后端会持续推送事件：
      - status: 任务状态更新
      - think: Agent 的思考过程
      - tool: 工具选择和执行
      - act: 动作执行
      - complete: 任务完成
      - error: 发生错误

    SSE 的工作原理：
      前端: const es = new EventSource('/tasks/xxx/events')
      es.addEventListener('think', (e) => { ... })
      后端: yield "event: think\ndata: {...}\n\n"  （这就是 SSE 格式）
    """
    async def event_generator():
        """
        异步生成器，持续从任务队列中读取事件并推送。
        这是一个无限循环，直到任务完成或客户端断开连接才停止。
        """
        # 如果任务不存在，发送错误事件并返回
        if task_id not in task_manager.queues:
            yield f"event: error\ndata: {dumps({'message': 'Task not found'})}\n\n"
            return

        queue = task_manager.queues[task_id]

        # 先发送当前状态（客户端刚连接时能看到已有进度）
        task = task_manager.tasks.get(task_id)
        if task:
            yield f"event: status\ndata: {dumps({'type': 'status', 'status': task.status, 'steps': task.steps})}\n\n"

        while True:
            try:
                # 从队列中读取事件（如果没有会等待）
                event = await queue.get()
                formatted_event = dumps(event)

                # 发送心跳包，保持连接不被超时关闭
                yield ": heartbeat\n\n"

                if event["type"] == "complete":
                    # 任务完成，发送完成事件并退出循环
                    yield f"event: complete\ndata: {formatted_event}\n\n"
                    break
                elif event["type"] == "error":
                    # 发生错误，发送错误事件并退出循环
                    yield f"event: error\ndata: {formatted_event}\n\n"
                    break
                elif event["type"] == "step":
                    # 步骤更新，同时发送状态和步骤事件
                    task = task_manager.tasks.get(task_id)
                    if task:
                        yield f"event: status\ndata: {dumps({'type': 'status', 'status': task.status, 'steps': task.steps})}\n\n"
                    yield f"event: {event['type']}\ndata: {formatted_event}\n\n"
                elif event["type"] in ["think", "tool", "act", "run"]:
                    # 思考/工具/动作/运行事件，直接推送
                    yield f"event: {event['type']}\ndata: {formatted_event}\n\n"
                else:
                    # 其他事件类型
                    yield f"event: {event['type']}\ndata: {formatted_event}\n\n"

            except asyncio.CancelledError:
                # 客户端断开连接时触发
                print(f"Client disconnected for task {task_id}")
                break
            except Exception as e:
                # 发生未知错误时推送错误事件并退出
                print(f"Error in event stream: {str(e)}")
                yield f"event: error\ndata: {dumps({'message': str(e)})}\n\n"
                break

    # 返回 StreamingResponse，将生成器作为 SSE 流返回
    # media_type="text/event-stream" 是 SSE 的标准 MIME 类型
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",      # 不缓存
            "Connection": "keep-alive",        # 保持长连接
            "X-Accel-Buffering": "no",         # 禁用 Nginx 缓冲（实时推送必须）
        },
    )


# ========== 任务查询接口 ==========

@app.get("/tasks")
async def get_tasks():
    """获取所有任务列表，按创建时间倒序排列"""
    sorted_tasks = sorted(
        task_manager.tasks.values(), key=lambda task: task.created_at, reverse=True
    )
    return JSONResponse(
        content=[task.model_dump() for task in sorted_tasks],
        headers={"Content-Type": "application/json"},
    )


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """获取单个任务的详细信息"""
    if task_id not in task_manager.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_manager.tasks[task_id]


# ========== 配置管理接口 ==========

@app.get("/config/status")
async def check_config_status():
    """检查配置文件状态，如果不存在则返回示例配置"""
    config_path = Path(__file__).parent / "config" / "config.toml"
    example_config_path = Path(__file__).parent / "config" / "config.example.toml"

    if config_path.exists():
        return {"status": "exists"}
    elif example_config_path.exists():
        try:
            with open(example_config_path, "rb") as f:
                example_config = tomllib.load(f)
            return {"status": "missing", "example_config": example_config}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "no_example"}


@app.post("/config/save")
async def save_config(config_data: dict = Body(...)):
    """保存配置文件，将前端提交的配置写入 config.toml"""
    try:
        config_dir = Path(__file__).parent / "config"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "config.toml"

        toml_content = ""

        if "llm" in config_data:
            toml_content += "# Global LLM configuration\n[llm]\n"
            llm_config = config_data["llm"]
            for key, value in llm_config.items():
                if key != "vision":
                    if isinstance(value, str):
                        toml_content += f'{key} = "{value}"\n'
                    else:
                        toml_content += f"{key} = {value}\n"

        if "server" in config_data:
            toml_content += "\n# Server configuration\n[server]\n"
            server_config = config_data["server"]
            for key, value in server_config.items():
                if isinstance(value, str):
                    toml_content += f'{key} = "{value}"\n'
                else:
                    toml_content += f"{key} = {value}\n"

        with open(config_path, "w", encoding="utf-8") as f:
            f.write(toml_content)

        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ========== 全局异常处理 ==========

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，捕获所有未处理的异常并返回 500 错误"""
    return JSONResponse(
        status_code=500, content={"message": f"Server error: {str(exc)}"}
    )


# ========== 辅助函数 ==========

def open_local_browser(config):
    """在默认浏览器中打开本地服务地址"""
    webbrowser.open_new_tab(f"http://{config['host']}:{config['port']}")


def load_config():
    """
    加载服务器配置（host 和 port）。
    如果配置文件不存在或缺少必要字段，则使用默认值。
    """
    try:
        config_path = Path(__file__).parent / "config" / "config.toml"

        # 配置文件不存在，使用默认值
        if not config_path.exists():
            return {"host": "localhost", "port": 5172}

        # 读取并解析 TOML 配置文件
        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        return {"host": config["server"]["host"], "port": config["server"]["port"]}
    except FileNotFoundError:
        # 文件未找到，返回默认值
        return {"host": "localhost", "port": 5172}
    except KeyError as e:
        # 配置文件缺少必要字段，返回默认值
        print(
            f"The configuration file is missing necessary fields: {str(e)}, use default configuration"
        )
        return {"host": "localhost", "port": 5172}


# ========== 程序入口 ==========
# 启动 Web 服务器
if __name__ == "__main__":
    import uvicorn  # ASGI 服务器，用于运行 FastAPI

    config = load_config()  # 加载配置（host 和 port）
    # 3 秒后自动打开浏览器（用 threading.Timer 延迟执行）
    open_with_config = partial(open_local_browser, config)
    threading.Timer(3, open_with_config).start()
    # 启动 Uvicorn 服务器
    uvicorn.run(app, host=config["host"], port=config["port"])
