"""
agent/base.py - Agent 抽象基类
================================
这是整个 Agent 继承体系的根节点，所有 Agent 都继承自这个类。

继承体系：
  BaseAgent (本文件)
    └── ReActAgent (react.py) - 定义 think() + act() 循环
          └── ToolCallAgent (toolcall.py) - 实现 LLM Function Calling
                └── Manus / BrowserAgent / SWEAgent / ...

BaseAgent 提供的核心功能：
1. 状态管理：IDLE → RUNNING → FINISHED/ERROR
2. 内存管理：保存对话历史（Memory）
3. 主循环：run() 方法循环调用 step()
4. 卡死检测：检测 Agent 是否陷入重复循环
"""

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """Agent 抽象基类

    为什么继承 BaseModel？
    Pydantic 的 BaseModel 提供数据验证和序列化能力，
    确保 Agent 的属性（如 llm、memory、state）始终合法。

    为什么继承 ABC？
    ABC 是抽象基类，强制子类必须实现 step() 方法。
    """

    # ---- 核心属性 ----
    name: str = Field(..., description="agent 的唯一名称")
    description: Optional[str] = Field(None, description="可选的 agent 描述")

    # ---- 提示词 ----
    system_prompt: Optional[str] = Field(
        None, description="系统级指令提示词（设定 Agent 的角色和行为规则）"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="提示 Agent 下一步应该做什么"
    )

    # ---- 依赖项 ----
    llm: LLM = Field(default_factory=LLM, description="语言模型实例（用于思考和决策）")
    memory: Memory = Field(default_factory=Memory, description="对话历史存储")
    state: AgentState = Field(
        default=AgentState.IDLE, description="当前状态：空闲/运行中/完成/错误"
    )

    # ---- 执行控制 ----
    max_steps: int = Field(default=10, description="最大执行步数（防止无限循环）")
    current_step: int = Field(default=0, description="当前执行到第几步")

    # 重复检测阈值：连续出现多少次相同回复就认为卡住了
    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # 允许额外字段，以便在子类中灵活使用

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """如果未提供，则使用默认设置初始化 agent。"""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """安全的状态转换上下文管理器

        用法示例：
            async with self.state_context(AgentState.RUNNING):
                # 在这个块内，self.state == RUNNING
                await self.step()
            # 退出后自动恢复之前的状态

        如果块内抛出异常，状态会自动设为 ERROR。
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # 失败时转换到 ERROR 状态
            raise e
        finally:
            self.state = previous_state  # 恢复到之前的状态

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """向 agent 的内存添加一条消息。

        Args:
            role: 消息发送者的角色（user, system, assistant, tool）。
            content: 消息内容。
            base64_image: 可选的 base64 编码图像。
            **kwargs: 额外参数（例如，工具消息的 tool_call_id）。

        Raises:
            ValueError: 如果角色不受支持。
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # 根据角色创建带有适当参数的消息
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Agent 的主执行循环（最重要的方法）

        执行流程：
        1. 检查状态必须是 IDLE
        2. 如果有用户请求，添加到内存
        3. 进入 RUNNING 状态
        4. 循环：step() → 检查是否卡住 → 记录结果
        5. 直到达到 max_steps 或状态变为 FINISHED
        6. 清理沙箱资源

        这就是 Agent 的“引擎”，所有具体 Agent 都通过这个循环来执行任务。
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"Executing step {self.current_step}/{self.max_steps}")
                # 执行一步：子类实现具体逻辑（think + act）
                step_result = await self.step()

                # 卡死检测：检查 Agent 是否在重复相同的回复
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"Step {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """执行单步操作（抽象方法，子类必须实现）

        在 ReActAgent 中，step() = think() + act()
        - think(): 调用 LLM 决定下一步做什么
        - act(): 执行工具调用或其他操作
        """

    def handle_stuck_state(self):
        """处理卡住状态

        当 Agent 连续多次返回相同的回复时，认为它卡住了。
        解决方式：在下一步的提示词中加入“请尝试新策略”的提示，
        引导 LLM 走出死循环。
        """
        stuck_prompt = "\
        观察到重复响应。请考虑新策略，避免重复已经尝试过的无效路径。"
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """检测 Agent 是否卡住

        检测方法：统计最近一条 assistant 消息的内容
        在历史消息中重复出现的次数。
        如果重复次数 >= duplicate_threshold，认为卡住了。
        """
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # 统计相同内容的出现次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """从 agent 的内存中检索消息列表。"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """设置 agent 内存中的消息列表。"""
        self.memory.messages = value
