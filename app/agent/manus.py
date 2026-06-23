"""
Manus Agent —— 项目的核心通用 Agent

这个文件定义了 Manus，它是整个项目最核心的 Agent，继承自 ToolCallAgent。
你可以把它理解为"一个什么都能干的 AI 助手"：
  - 它能执行 Python 代码（PythonExecute）
  - 它能操作浏览器（BrowserUseTool）
  - 它能搜索网页（WebSearch）
  - 它能编辑文件（StrReplaceEditor）
  - 它能向人类提问（AskHuman）
  - 它还能连接外部 MCP 工具服务器（MCPClients）

继承关系：
  BaseAgent → ReActAgent → ToolCallAgent → Manus

核心特点：
  1. 工具集合丰富：默认包含 6 种常用工具
  2. MCP 协议支持：可以动态连接/断开外部工具服务器
  3. 浏览器上下文感知：如果浏览器工具被使用，会自动格式化包含页面元素的提示词
  4. 智能模型切换：浏览器操作时可根据需要切换模型
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper  # 浏览器上下文辅助工具
from app.agent.toolcall import ToolCallAgent        # 父类：工具调用 Agent
from app.config import config                      # 全局配置
from app.llm import LLM                            # 大语言模型封装
from app.logger import logger                      # 日志工具
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT  # 系统提示词模板
from app.schema import Message                     # 消息数据结构
from app.tool import Terminate, ToolCollection     # 终止工具 + 工具集合管理器
from app.tool.ask_human import AskHuman            # 向人类提问的工具
from app.tool.browser_use_tool import BrowserUseTool  # 浏览器操作工具
from app.tool.mcp import MCPClients, MCPClientTool    # MCP 远程工具协议
from app.tool.python_execute import PythonExecute     # Python 代码执行工具
from app.tool.str_replace_editor import StrReplaceEditor  # 文件编辑工具
from app.tool.web_search import WebSearch              # 网页搜索工具


class Manus(ToolCallAgent):
    """
    Manus —— 一个通用的多功能 Agent。

    这是整个项目最核心的类，相当于一个"万能 AI 助手"。
    它继承了 ToolCallAgent 的所有能力（think→act 循环、工具调用），
    并且在此基础上增加了：
      - 丰富的内置工具集（代码执行、浏览器、搜索、文件编辑等）
      - MCP 远程工具服务器的连接与管理
      - 浏览器上下文感知（自动获取浏览器页面元素信息）
      - 智能模型切换策略
    """

    # Agent 的名字和描述，用于在多 Agent 协作时标识自己
    name: str = "Manus"
    description: str = "一个多功能的 agent，可以使用多种工具（包括基于 MCP 的工具）解决各种任务"

    # system_prompt: 系统提示词，告诉 LLM "你是谁、你能做什么"
    # 使用 format() 注入两个动态信息：
    #   - {directory}: 当前工作目录路径（从配置文件读取）
    #   - {current_datetime}: 当前时间（中文格式，如"2024年01月15日 14:30:00 Monday"）
    # 这样 LLM 就知道当前在哪个目录工作，以及当前时间
    system_prompt: str = SYSTEM_PROMPT.format(
        directory=config.workspace_root,
        current_datetime=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S %A"),
    )
    # next_step_prompt: 每一步决策时的额外提示，引导 LLM 如何行动
    next_step_prompt: str = NEXT_STEP_PROMPT

    # max_observe: 工具返回结果的最大观察长度（超过会被截断），防止上下文太长
    max_observe: int = 10000
    # max_steps: 最多执行 20 步（think→act 循环），超过就强制停止，防止死循环
    max_steps: int = 20

    # ========== MCP 相关 ==========
    # MCP（Model Context Protocol）是一种让 AI 连接外部工具服务器的协议。
    # 想象一下：你有一个远程服务器，上面跑着数据库查询工具、API 调用工具等，
    # Manus 可以通过 MCP 协议连接到这些服务器，动态获取并使用它们的工具。
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # ========== 内置工具集合 ==========
    # ToolCollection 是一个"工具箱"，里面存放所有 Agent 可以使用的工具。
    # 当 LLM 决定要使用某个工具时，会从这个工具箱中查找对应的工具并执行。
    # 默认包含 6 个工具：
    #   1. PythonExecute  - 执行 Python 代码（用多进程隔离，安全）
    #   2. BrowserUseTool - 操控浏览器（点击、输入、截图等）
    #   3. WebSearch      - 网页搜索（支持多个搜索引擎）
    #   4. StrReplaceEditor - 文件编辑器（查找替换文件内容）
    #   5. AskHuman       - 向人类提问（当 AI 需要帮助时）
    #   6. Terminate      - 终止工具（任务完成时调用，结束循环）
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(),
            BrowserUseTool(),
            WebSearch(),
            StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    # special_tool_names: 特殊工具列表，这些工具被调用时会触发特殊行为
    # Terminate 就是特殊工具 —— 一旦被调用，Agent 会停止运行
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])

    # browser_context_helper: 浏览器上下文辅助工具
    # 负责获取浏览器当前状态（URL、页面元素列表等），并格式化到提示词中
    browser_context_helper: Optional[BrowserContextHelper] = None

    # connected_servers: 记录已连接的 MCP 服务器 {服务器ID: 地址}
    # 用于跟踪哪些服务器已连接，方便后续断开连接
    connected_servers: Dict[str, str] = Field(default_factory=dict)

    # _initialized: 标记 Agent 是否已完成初始化（MCP 服务器是否已连接）
    _initialized: bool = False

    @model_validator(mode="after")
    def initialize_helper(self) -> "Manus":
        """
        Pydantic 模型验证器，在模型创建后自动执行。
        mode="after" 表示在 Pydantic 完成所有字段验证后再执行此方法。

        这里做的事情：创建 BrowserContextHelper 实例。
        为什么不在 __init__ 里创建？因为 BrowserContextHelper 需要引用 self（Agent 本身），
        而 Pydantic 的模型初始化过程中 self 可能还没完全准备好。
        """
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "Manus":
        """
        工厂方法 —— 创建并正确初始化 Manus 实例。

        为什么不直接用 Manus() 创建？
        因为 MCP 服务器的连接是异步操作（async），而 __init__ 不能是 async 的。
        所以我们用这个 async 的工厂方法来：
          1. 先创建实例（同步）
          2. 再连接 MCP 服务器（异步）
          3. 标记初始化完成

        使用示例：
            agent = await Manus.create()  # 创建并初始化
            await agent.run("帮我搜索今天的新闻")  # 运行
        """
        instance = cls(**kwargs)                    # 第 1 步：创建实例
        await instance.initialize_mcp_servers()      # 第 2 步：连接 MCP 服务器
        instance._initialized = True                 # 第 3 步：标记初始化完成
        return instance

    async def initialize_mcp_servers(self) -> None:
        """
        初始化与已配置的 MCP 服务器的连接。

        从配置文件（config.toml）中读取 MCP 服务器列表，
        然后逐个连接。支持两种连接方式：
          - SSE（Server-Sent Events）：通过 HTTP URL 连接，适合远程服务器
          - stdio（标准输入输出）：通过启动本地进程连接，适合本地工具

        即使某个服务器连接失败，也不会影响其他服务器的连接（try-except 隔离错误）。
        """
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"Connected to MCP server {server_id} at {server_config.url}"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"Connected to MCP server {server_id} using command {server_config.command}"
                        )
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {server_id}: {e}")

    async def connect_mcp_server(
        self,
        server_url: str,        # 服务器地址（SSE 模式下是 URL，stdio 模式下是命令路径）
        server_id: str = "",    # 服务器唯一标识符
        use_stdio: bool = False, # 是否使用 stdio 模式（默认使用 SSE）
        stdio_args: List[str] = None,  # stdio 模式的启动参数
    ) -> None:
        """
        连接到 MCP 服务器并添加其工具。

        连接成功后，该服务器提供的所有工具会自动注册到 available_tools 中，
        这样 Agent 在下一步决策时就可以选择使用这些远程工具了。
        """
        if use_stdio:
            # stdio 模式：启动本地进程，通过标准输入输出通信
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            # SSE 模式：通过 HTTP URL 连接到远程服务器
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # 从 MCP 客户端的所有工具中，筛选出属于当前服务器的工具，添加到工具箱
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """
        断开与 MCP 服务器的连接并移除其工具。

        为什么要重建工具列表？
        因为 available_tools 里既有本地工具（如 PythonExecute），也有远程工具（MCPClientTool）。
        断开服务器后，需要把属于该服务器的远程工具从工具箱中移除，
        但保留所有本地工具和其他服务器的工具。
        """
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # 第 1 步：从工具箱中筛选出所有非 MCP 工具（即本地工具）
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        # 第 2 步：用本地工具重建工具箱
        self.available_tools = ToolCollection(*base_tools)
        # 第 3 步：把仍然连接的 MCP 服务器工具重新加入
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def cleanup(self):
        """
        清理 Manus Agent 的资源。

        在 Agent 完成工作或程序退出前调用，确保：
          1. 浏览器被关闭（释放 Chromium 进程）
          2. 所有 MCP 服务器连接被断开（释放网络连接）
        """
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # 仅在已完成初始化的情况下才断开 MCP 服务器连接
        if self._initialized:
            await self.disconnect_mcp_server()
            self._initialized = False

    async def think(self) -> bool:
        """
        Manus 的"思考"方法 —— 重写父类 ToolCallAgent 的 think()。

        这是整个 Manus 最复杂的方法，它在父类 think() 的基础上增加了：
          1. MCP 服务器延迟初始化（第一次思考时才连接）
          2. 浏览器上下文感知（获取页面元素列表，注入到提示词中）
          3. 智能模型切换策略

        思考流程：
          1. 如果还没初始化，先连接 MCP 服务器
          2. 检查最近 5 条消息，判断浏览器是否正在使用
          3. 如果浏览器工具可用，用 BrowserContextHelper 格式化提示词
             （这样 LLM 能看到当前页面的 URL、可交互元素列表等信息）
          4. 调用父类的 think() 让 LLM 做决策
          5. 恢复原始提示词和模型

        Returns:
            bool: True 表示 LLM 成功做出了决策，False 表示需要停止
        """
        # 延迟初始化：第一次 think 时才连接 MCP 服务器
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        # 保存原始的提示词和 LLM，方便后面恢复
        original_prompt = self.next_step_prompt
        original_llm = self.llm

        # ===== 浏览器状态检测 =====
        # 检查最近的 5 条消息，看看浏览器工具是否正在被使用
        recent_messages = self.memory.messages[-5:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == BrowserUseTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        # 检查消息中是否有浏览器截图（base64 编码的图片）
        has_browser_screenshot = any(
            (isinstance(msg, Message) and msg.base64_image)
            or (isinstance(msg, dict) and msg.get("base64_image"))
            for msg in recent_messages
        )

        # 检查工具列表中是否包含浏览器工具
        browser_tool_available = BrowserUseTool().name in [
            tool.name for tool in self.available_tools.tools
        ]

        # ===== 智能切换策略 =====
        # browser-use 库会把页面上的可交互元素格式化为：[index]<type>text</type>
        # 例如：[3]<button>搜索</button> 表示索引为 3 的按钮，文字是"搜索"
        # 这些文本描述通常足够详细，LLM 可以根据文本匹配选择正确的元素
        # 所以默认使用普通模型（如 qwen-max），不需要切换到更贵的视觉模型

        # 如果浏览器工具可用，用 BrowserContextHelper 格式化提示词
        if browser_tool_available:
            logger.debug(f"🚀 Using default model for browser automation: {self.llm.model}")
            logger.debug(f"📝 Browser-use provides element descriptions in format: [index]<type>text</type>")

            # 用 BrowserContextHelper 获取浏览器状态并格式化到提示词中
            # 这样 LLM 能看到：当前 URL、页面标题、可交互元素列表等
            # 即使浏览器还没打开，也会提示需要打开
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        # 调用父类的 think()，让 LLM 根据当前状态做决策
        result = await super().think()

        # ===== 恢复原始状态 =====
        # 恢复原始提示词（下次 think 时会重新生成）
        self.next_step_prompt = original_prompt
        # 如果没有浏览器截图，并且 LLM 被切换过，就恢复到原始模型
        if not has_browser_screenshot and original_llm != self.llm:
            # 再检查最近 3 条消息，确认浏览器已不再使用
            current_browser_in_use = any(
                tc.function.name == BrowserUseTool().name
                for msg in self.memory.messages[-3:]
                if hasattr(msg, 'tool_calls') and msg.tool_calls
                for tc in msg.tool_calls
            )
            # 再检查最近 3 条消息，确认没有截图
            current_has_screenshot = any(
                (isinstance(msg, Message) and msg.base64_image)
                or (isinstance(msg, dict) and msg.get("base64_image"))
                for msg in self.memory.messages[-3:]
            )
            # 如果浏览器不再使用且没有截图，恢复原始（更快更便宜）模型
            if not current_browser_in_use and not current_has_screenshot:
                logger.debug(f"🔄 Restoring original LLM: {original_llm.model}")
                self.llm = original_llm

        return result
