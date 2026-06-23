"""
tool/tool_collection.py - 工具集合管理
=======================================
管理多个工具的集合，类似“工具箱”。

核心功能：
- 存储多个 BaseTool 实例
- to_params(): 将所有工具转换为 OpenAI Function Calling 格式（发给 LLM）
- execute(): 根据工具名称查找并执行对应工具
- add_tool/add_tools: 动态添加工具（如连接 MCP 服务器后获得新工具）

调用链路：
  Agent.think() → available_tools.to_params() → 发给 LLM
  Agent.act() → available_tools.execute(name, args) → 执行工具
"""

from typing import Any, Dict, List

from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """工具集合（工具箱）

    将多个工具组织在一起，提供统一的接口给 Agent 使用。

    内部结构：
    - tools: 元组，存储所有工具实例
    - tool_map: 字典，按名称快速查找工具（name → tool）
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool):
        """初始化工具集合

        Args:
            *tools: 任意数量的 BaseTool 实例
            例如: ToolCollection(PythonExecute(), WebSearch(), Terminate())
        """
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def __iter__(self):
        return iter(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        """将所有工具转换为 OpenAI Function Calling 格式

        返回的列表会直接传给 LLM，让 LLM 知道有哪些工具可用。
        每个元素就是调用 tool.to_param() 的结果。
        """
        return [tool.to_param() for tool in self.tools]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        """根据工具名称执行对应工具

        流程：
        1. 从 tool_map 中按名称查找工具
        2. 找不到则返回 ToolFailure
        3. 找到则调用工具的 __call__ 方法（实际执行 execute()）
        """
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """按顺序执行集合中的所有工具。"""
        results = []
        for tool in self.tools:
            try:
                result = await tool()
                results.append(result)
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
        return results

    def get_tool(self, name: str) -> BaseTool:
        return self.tool_map.get(name)

    def add_tool(self, tool: BaseTool):
        """向集合中添加单个工具。

        如果已存在同名工具，将跳过并记录警告。
        """
        if tool.name in self.tool_map:
            logger.warning(f"Tool {tool.name} already exists in collection, skipping")
            return self

        self.tools += (tool,)
        self.tool_map[tool.name] = tool
        return self

    def add_tools(self, *tools: BaseTool):
        """向集合中添加多个工具。

        如果任何工具与现有工具存在名称冲突，将跳过并记录警告。
        """
        for tool in tools:
            self.add_tool(tool)
        return self
