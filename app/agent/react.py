"""
agent/react.py - ReAct Agent 抽象类
=====================================
实现 ReAct（Reasoning + Acting）模式的中间层。

什么是 ReAct？
ReAct 是一种让 LLM 交替“思考”和“行动”的模式：
- think(): 思考阶段 - 调用 LLM 分析当前情况，决定下一步做什么
- act():   行动阶段 - 执行工具调用或其他操作

一步 step() = think() + act()，循环执行直到任务完成。

这个类本身是抽象的，不直接使用。
它的子类 ToolCallAgent 实现了具体的 think() 和 act() 逻辑。
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    """ReAct Agent 抽象类

    在 BaseAgent 的基础上定义了 think-act 循环：
    - think(): 抽象方法，子类实现“思考”逻辑
    - act():   抽象方法，子类实现“行动”逻辑
    - step():  = think() → act()
    """
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """思考阶段（抽象方法）

        子类实现具体逻辑，通常是：
        1. 将对话历史发给 LLM
        2. LLM 返回下一步的决策（调用哪个工具、传什么参数）
        3. 返回 True 表示需要执行行动，False 表示无需行动
        """

    @abstractmethod
    async def act(self) -> str:
        """行动阶段（抽象方法）

        子类实现具体逻辑，通常是：
        1. 遍历 think() 中决定的工具调用
        2. 逐个执行工具
        3. 将结果添加到内存
        4. 返回执行结果的文本摘要
        """

    async def step(self) -> str:
        """执行一步：先思考，再行动

        这就是 ReAct 的核心：每一步都是 think → act 的循环。
        如果 think() 返回 False（不需要行动），则跳过 act()。
        """
        should_act = await self.think()
        if not should_act:
            return "思考完成 - 无需行动"
        return await self.act()
