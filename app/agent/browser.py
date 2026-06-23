"""
agent/browser.py —— 浏览器相关的 Agent 和辅助工具

这个文件包含两部分：

1. BrowserContextHelper（浏览器上下文辅助工具）
   - 获取浏览器当前状态（URL、页面标题、可交互元素列表）
   - 将状态格式化到提示词中，让 LLM 知道当前页面的情况
   - 清理浏览器资源

2. BrowserAgent（浏览器专用 Agent）
   - 继承自 ToolCallAgent，专门用于操作浏览器
   - 工具集只有 BrowserUseTool + Terminate
   - 每次 think() 时自动获取浏览器状态并注入提示词

关键设计决策：
  - 只使用文本元素描述，不发送截图给 LLM
  - 因为 browser-use 库返回的元素信息已足够详细：
    [index]<type>text</type> （如 [3]<button>搜索</button>）
  - 这样可以用普通模型（如 qwen-max），不需要更贵的视觉模型
"""

import json
from typing import TYPE_CHECKING, Optional

from pydantic import Field, model_validator

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.browser import NEXT_STEP_PROMPT, SYSTEM_PROMPT  # 浏览器专用提示词
from app.schema import Message, ToolChoice
from app.tool import BrowserUseTool, Terminate, ToolCollection
from app.tool.sandbox.sb_browser_tool import SandboxBrowserTool  # 沙箱环境浏览器工具


# TYPE_CHECKING 只在类型检查时为 True，运行时为 False
# 这样可以避免循环导入问题（browser.py 和 base.py 相互引用）
if TYPE_CHECKING:
    from app.agent.base import BaseAgent


class BrowserContextHelper:
    """
    浏览器上下文辅助工具。

    职责：
      1. 获取浏览器当前状态（URL、标题、可交互元素列表、截图）
      2. 将状态格式化为提示词，让 LLM 知道页面当前是什么样子
      3. 清理浏览器资源

    使用场景：
      - Manus Agent 的 think() 方法中调用 format_next_step_prompt()
      - BrowserAgent 的 think() 方法中也调用
    """

    def __init__(self, agent: "BaseAgent"):
        self.agent = agent  # 关联的 Agent 实例
        self._current_base64_image: Optional[str] = None  # 当前页面截图（base64）

    async def get_browser_state(self) -> Optional[dict]:
        """
        获取浏览器当前状态。

        Returns:
            包含 URL、标题、可交互元素等信息的字典，如果浏览器未就绪返回 None
        """
        # 尝试从工具集中获取浏览器工具
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if not browser_tool:
            # 如果普通浏览器工具不存在，尝试沙箱浏览器工具
            browser_tool = self.agent.available_tools.get_tool(
                SandboxBrowserTool().name
            )
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            logger.warning("BrowserUseTool not found or doesn't have get_current_state")
            return None
        try:
            # 获取浏览器状态并保存截图
            result = await browser_tool.get_current_state()
            if result.error:
                logger.debug(f"Browser state error: {result.error}")
                return None
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image
            else:
                self._current_base64_image = None
            return json.loads(result.output)
        except Exception as e:
            logger.debug(f"Failed to get browser state: {str(e)}")
            return None

    async def format_next_step_prompt(self) -> str:
        """
        获取浏览器状态并格式化为提示词。

        返回的提示词包含：
          - 当前 URL 和页面标题
          - 标签页数量
          - 可滚动区域信息
          - 可交互元素列表（关键！LLM 根据这个选择点击哪个元素）
        """
        browser_state = await self.get_browser_state()
        url_info, tabs_info, content_above_info, content_below_info = "", "", "", ""
        results_info = ""  # 或者如果需要，从 agent 获取

        if browser_state and not browser_state.get("error"):
            # 提取 URL 和标题信息
            url_info = f"\n   URL: {browser_state.get('url', 'N/A')}\n   Title: {browser_state.get('title', 'N/A')}"
            tabs = browser_state.get("tabs", [])
            if tabs:
                tabs_info = f"\n   {len(tabs)} tab(s) available"
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)
            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"
            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            # 调试信息：显示可交互元素数量
            interactive_elements = browser_state.get("interactive_elements", "")
            element_count = interactive_elements.count("[") if interactive_elements else 0
            logger.info(f"🔍 Browser state: {element_count} interactive elements detected")
            logger.debug(f"🔍 Browser URL: {browser_state.get('url', 'N/A')}")
            logger.debug(f"🔍 Browser Title: {browser_state.get('title', 'N/A')}")
            if interactive_elements:
                preview = interactive_elements[:200] + "..." if len(interactive_elements) > 200 else interactive_elements
                logger.debug(f"🔍 Interactive elements preview: {preview}")

            # 截图处理：即使有截图也丢弃，只用文本元素描述
            # 这样可以用更便宜的模型，不需要视觉理解
            if self._current_base64_image:
                image_size_kb = len(self._current_base64_image) * 3 / 4 / 1024  # 估算图片大小（KB）
                logger.debug(f"📸 Browser screenshot captured: {image_size_kb:.2f} KB (base64) - but not sending to LLM")
                logger.debug(f"📝 Using element text descriptions instead of visual model")
                # 不发送截图，只使用文本元素描述
                self._current_base64_image = None  # 丢弃截图，不使用视觉模型
            else:
                logger.debug("📝 No screenshot - using element text descriptions only")

        # 构建完整提示词，用占位符替换动态信息
        prompt = NEXT_STEP_PROMPT.format(
            url_placeholder=url_info,
            tabs_placeholder=tabs_info,
            content_above_placeholder=content_above_info,
            content_below_placeholder=content_below_info,
            results_placeholder=results_info,
        )

        # 将可交互元素列表附加到提示词末尾（这是关键！）
        # LLM 会根据这个列表选择要点击/输入的元素
        if browser_state and not browser_state.get("error"):
            interactive_elements = browser_state.get("interactive_elements", "")
            if interactive_elements:
                prompt += "\n\n[Current state starts here]\n"
                prompt += "Interactive Elements:\n"
                prompt += interactive_elements
                prompt += "\n"

        return prompt

    async def cleanup_browser(self):
        browser_tool = self.agent.available_tools.get_tool(BrowserUseTool().name)
        if browser_tool and hasattr(browser_tool, "cleanup"):
            await browser_tool.cleanup()


class BrowserAgent(ToolCallAgent):
    """
    浏览器专用 Agent —— 专门用于操作浏览器完成任务。

    与 Manus 的区别：
      - Manus 是全能 Agent，包含多种工具
      - BrowserAgent 只有浏览器工具，更轻量，适合纯浏览器任务

    继承关系：
      BaseAgent → ReActAgent → ToolCallAgent → BrowserAgent
    """

    name: str = "browser"
    description: str = "可以控制浏览器来完成任务的浏览器 agent"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # 工具集：只有浏览器工具 + 终止工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(BrowserUseTool(), Terminate())
    )

    tool_choices: ToolChoice = ToolChoice.AUTO  # 自动决定是否使用工具
    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])  # 终止工具

    browser_context_helper: Optional[BrowserContextHelper] = None

    @model_validator(mode="after")
    def initialize_helper(self) -> "BrowserAgent":
        """创建浏览器上下文辅助工具"""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    async def think(self) -> bool:
        """
        重写 think()，每次思考前自动获取浏览器状态。
        这样 LLM 始终知道当前页面的 URL、元素列表等信息。
        """
        self.next_step_prompt = (
            await self.browser_context_helper.format_next_step_prompt()
        )
        return await super().think()

    async def cleanup(self):
        """清理浏览器资源（关闭 Chromium 进程）"""
        await self.browser_context_helper.cleanup_browser()
