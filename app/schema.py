"""
schema.py - 项目核心数据结构定义
=================================
这个文件定义了整个 OpenManus 项目中最基础的数据结构，包括：
- Role: 消息角色（谁说的话）
- ToolChoice: 工具选择策略（是否允许使用工具）
- AgentState: Agent 的运行状态
- Function / ToolCall: 工具调用的数据结构
- Message: 对话中的单条消息
- Memory: Agent 的消息记忆（对话历史）

这些数据结构被项目中几乎所有模块引用，是理解项目的第一步。
"""

from enum import Enum
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ============================================================
# 枚举类型定义
# ============================================================

class Role(str, Enum):
    """消息角色选项

    在 LLM 对话中，每条消息都有一个角色，表示这条消息是谁发出的：
    - SYSTEM: 系统消息，用于设定 AI 的行为规则（类似"人设"）
    - USER: 用户消息，用户输入的提问或指令
    - ASSISTANT: 助手消息，AI 的回复内容
    - TOOL: 工具消息，工具执行后返回的结果
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# 将枚举值转为元组，用于 Literal 类型约束（确保字段只能取这些值）
ROLE_VALUES = tuple(role.value for role in Role)
ROLE_TYPE = Literal[ROLE_VALUES]  # type: ignore  # 等价于 Literal["system", "user", "assistant", "tool"]


class ToolChoice(str, Enum):
    """工具选择策略

    控制 LLM 是否可以使用工具：
    - NONE: 禁止使用工具，LLM 只能纯文本回复
    - AUTO: 自动决定，LLM 可以选择用工具或不用（最常用）
    - REQUIRED: 强制使用工具，LLM 必须调用至少一个工具
    """

    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore  # 等价于 Literal["none", "auto", "required"]


class AgentState(str, Enum):
    """Agent 的运行状态（状态机）

    Agent 在生命周期中会在这些状态之间转换：
    - IDLE: 空闲状态，等待被启动
    - RUNNING: 正在执行任务（think-act 循环中）
    - FINISHED: 任务完成（可能是正常完成或被 Terminate 工具终止）
    - ERROR: 执行过程中发生了错误

    状态转换流程：IDLE → RUNNING → FINISHED/ERROR → IDLE
    """

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


# ============================================================
# 工具调用相关数据结构
# ============================================================

class Function(BaseModel):
    """函数调用的详细信息

    当 LLM 决定调用一个工具时，会返回：
    - name: 要调用的工具名称（如 "web_search"、"python_execute"）
    - arguments: 传给工具的参数（JSON 格式的字符串，如 '{"query": "天气"}')
    """
    name: str
    arguments: str


class ToolCall(BaseModel):
    """表示一次工具调用

    这是 OpenAI Function Calling 标准格式的数据结构：
    - id: 这次工具调用的唯一标识（用于将结果匹配回对应的调用）
    - type: 调用类型，固定为 "function"
    - function: 具体的函数信息（名称 + 参数）

    一次 LLM 回复可能包含多个 ToolCall（同时调用多个工具）
    """

    id: str
    type: str = "function"
    function: Function


# ============================================================
# 消息与记忆（对话历史）
# ============================================================

class Message(BaseModel):
    """表示对话中的一条聊天消息

    这是项目中最核心的数据结构之一，贯穿整个 Agent 的执行过程。
    每条消息包含：
    - role: 消息角色（system/user/assistant/tool）
    - content: 消息的文本内容
    - tool_calls: 如果是 assistant 消息，可能包含工具调用请求
    - base64_image: 可选的图片数据（用于浏览器截图等视觉信息）
    """

    role: ROLE_TYPE = Field(...)  # type: ignore  # 消息角色（必填）
    content: Optional[str] = Field(default=None)  # 消息文本内容
    tool_calls: Optional[List[ToolCall]] = Field(default=None)  # 工具调用列表
    name: Optional[str] = Field(default=None)  # 工具名称（仅 tool 角色消息使用）
    tool_call_id: Optional[str] = Field(default=None)  # 对应的工具调用 ID
    base64_image: Optional[str] = Field(default=None)  # base64 编码的图片（如浏览器截图）

    def __add__(self, other) -> List["Message"]:
        """支持 Message + list 或 Message + Message 的操作"""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other) -> List["Message"]:
        """支持 list + Message 的操作"""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> dict:
        """将消息转换为字典格式

        用于发送给 OpenAI API，因为 API 接收的是字典而不是 Pydantic 对象。
        只添加非 None 的字段，避免发送多余的空字段。
        """
        message = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.dict() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        return message

    # ---- 工厂方法：快速创建不同类型的消息 ----
    # 这些是便捷方法，避免每次创建消息都要手动指定 role

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建用户消息（用户说的话）"""
        return cls(role=Role.USER, content=content, base64_image=base64_image)

    @classmethod
    def system_message(cls, content: str) -> "Message":
        """创建系统消息（AI 的行为规则/人设）"""
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None
    ) -> "Message":
        """创建助手消息（AI 的回复）"""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image)

    @classmethod
    def tool_message(
        cls, content: str, name, tool_call_id: str, base64_image: Optional[str] = None
    ) -> "Message":
        """创建工具消息（工具执行后返回的结果）"""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs,
    ):
        """从原始工具调用创建 ToolCallsMessage。

        Args:
            tool_calls: 来自 LLM 的原始工具调用
            content: 可选的消息内容
            base64_image: 可选的 base64 编码图像
        """
        formatted_calls = [
            {"id": call.id, "function": call.function.model_dump(), "type": "function"}
            for call in tool_calls
        ]
        return cls(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    """Agent 的消息记忆（对话历史）

    Memory 本质上就是一个消息列表，加上一个最大消息数限制。
    它的作用是保存 Agent 执行过程中的所有对话记录，
    这样 Agent 在每一步 think() 时能看到之前的上下文。

    为什么需要 max_messages？
    LLM 有上下文窗口限制（如 128K tokens），如果对话太长会超出限制。
    通过限制最大消息数，自动丢弃最早的消息来控制长度（滑动窗口）。
    """
    messages: List[Message] = Field(default_factory=list)  # 消息列表
    max_messages: int = Field(default=100)  # 最多保留 100 条消息

    def add_message(self, message: Message) -> None:
        """向内存添加一条消息"""
        self.messages.append(message)
        # 可选：实现消息限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def add_messages(self, messages: List[Message]) -> None:
        """向内存添加多条消息"""
        self.messages.extend(messages)
        # 可选：实现消息限制
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def clear(self) -> None:
        """清除所有消息"""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """获取最近的 n 条消息"""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """将消息转换为字典列表"""
        return [msg.to_dict() for msg in self.messages]
