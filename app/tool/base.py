"""
tool/base.py - 工具基类和结果封装
===================================
这个文件定义了所有工具的“模板”，是理解工具系统的第一步。

核心内容：
1. ToolResult: 工具执行结果的封装（输出/错误/图片）
2. BaseTool: 所有工具的抽象基类（定义了工具必须实现的接口）
3. CLIResult / ToolFailure: ToolResult 的特化子类

设计思路：
- 每个工具继承 BaseTool，实现 execute() 方法
- 工具执行后返回 ToolResult，统一处理成功/失败/图片等情况
- to_param() 将工具转换为 OpenAI Function Calling 的 JSON Schema 格式
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from app.utils.logger import logger


# class BaseTool(ABC, BaseModel):
#     name: str
#     description: str
#     parameters: Optional[dict] = None

#     class Config:
#         arbitrary_types_allowed = True

#     async def __call__(self, **kwargs) -> Any:
#         """Execute the tool with given parameters."""
#         return await self.execute(**kwargs)

#     @abstractmethod
#     async def execute(self, **kwargs) -> Any:
#         """Execute the tool with given parameters."""

#     def to_param(self) -> Dict:
#         """Convert tool to function call format."""
#         return {
#             "type": "function",
#             "function": {
#                 "name": self.name,
#                 "description": self.description,
#                 "parameters": self.parameters,
#             },
#         }


class ToolResult(BaseModel):
    """工具执行结果的封装

    每个工具执行后都会返回一个 ToolResult，包含：
    - output: 成功时的输出内容（文本或任意类型）
    - error: 失败时的错误信息
    - base64_image: 可选的图片（如浏览器截图）
    - system: 系统提示信息

    __bool__ 方法：只有当任意字段有值时，ToolResult 才被视为 True
    __str__ 方法：如果有错误则显示错误，否则显示输出
    """

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        return f"Error: {self.error}" if self.error else self.output

    def replace(self, **kwargs):
        """返回一个替换了给定字段的新 ToolResult。"""
        # return self.copy(update=kwargs)
        return type(self)(**{**self.dict(), **kwargs})


class BaseTool(ABC, BaseModel):
    """所有工具的抽象基类

    任何新工具都必须继承这个类，并实现 execute() 方法。
    继承关系：BaseTool(ABC, BaseModel)
    - ABC: 抽象基类，强制子类实现 execute()
    - BaseModel: Pydantic 模型，提供数据验证

    子类必须定义的属性：
    - name: 工具名称（如 "web_search"）
    - description: 工具描述（LLM 根据这个描述决定是否使用工具）
    - parameters: 参数的 JSON Schema（告诉 LLM 工具接受哪些参数）

    核心方法：
    - execute(): 实际执行逻辑（子类实现）
    - to_param(): 转换为 OpenAI Function Calling 格式
    """

    name: str
    description: str
    parameters: Optional[dict] = None
    # _schemas: Dict[str, List[ToolSchema]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # def __init__(self, **data):
    #     """Initialize tool with model validation and schema registration."""
    #     super().__init__(**data)
    #     logger.debug(f"Initializing tool class: {self.__class__.__name__}")
    #     self._register_schemas()

    # def _register_schemas(self):
    #     """Register schemas from all decorated methods."""
    #     for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
    #         if hasattr(method, 'tool_schemas'):
    #             self._schemas[name] = method.tool_schemas
    #             logger.debug(f"Registered schemas for method '{name}' in {self.__class__.__name__}")

    async def __call__(self, **kwargs) -> Any:
        """使用给定参数执行工具。"""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """使用给定参数执行工具。"""

    def to_param(self) -> Dict:
        """将工具转换为 OpenAI Function Calling 格式

        返回的字典结构如下（这是 OpenAI API 要求的格式）：
        {
            "type": "function",
            "function": {
                "name": "web_search",           # 工具名称
                "description": "搜索网页...",   # 工具描述
                "parameters": {...}             # 参数的 JSON Schema
            }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # def get_schemas(self) -> Dict[str, List[ToolSchema]]:
    #     """Get all registered tool schemas.

    #     Returns:
    #         Dict mapping method names to their schema definitions
    #     """
    #     return self._schemas

    def success_response(self, data: Union[Dict[str, Any], str]) -> ToolResult:
        """创建成功的工具结果。

        Args:
            data: 结果数据（字典或字符串）

        Returns:
            带有 success=True 和格式化输出的 ToolResult
        """
        if isinstance(data, str):
            text = data
        else:
            text = json.dumps(data, indent=2)
        logger.debug(f"Created success response for {self.__class__.__name__}")
        return ToolResult(output=text)

    def fail_response(self, msg: str) -> ToolResult:
        """创建失败的工具结果。

        Args:
            msg: 描述失败的错误消息

        Returns:
            带有 success=False 和错误消息的 ToolResult
        """
        logger.debug(f"Tool {self.__class__.__name__} returned failed result: {msg}")
        return ToolResult(error=msg)


class CLIResult(ToolResult):
    """可以渲染为 CLI 输出的 ToolResult。"""


class ToolFailure(ToolResult):
    """表示失败的 ToolResult。"""
