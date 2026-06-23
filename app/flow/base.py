"""
flow/base.py —— 流程编排的基类

这个文件定义了 BaseFlow，它是所有“流程”的父类。
所谓“流程”，就是让多个 Agent 协作完成任务的编排逻辑。

继承关系：
  BaseModel + ABC → BaseFlow → PlanningFlow

BaseFlow 的核心职责：
  1. 管理多个 Agent（以字典形式存储）
  2. 指定一个"主 Agent"（primary_agent）
  3. 定义 execute() 抽象方法，由子类实现具体的执行逻辑

类比理解：
  BaseFlow 就像一个“项目经理”，它自己不干活，
  但它管理着多个“员工”（Agent），并决定谁去做什么。
"""

from abc import ABC, abstractmethod      # 抽象基类
from typing import Dict, List, Optional, Union

from pydantic import BaseModel           # 数据模型基类

from app.agent.base import BaseAgent     # Agent 基类


class BaseFlow(BaseModel, ABC):
    """
    流程编排的基类，支持多个 Agent 协作。

    属性：
      - agents: Agent 字典，key 是名称（如 "manus"），value 是 Agent 实例
      - tools: 可选的工具列表（流程级别的工具）
      - primary_agent_key: 主 Agent 在 agents 字典中的 key
    """

    agents: Dict[str, BaseAgent]              # Agent 字典
    tools: Optional[List] = None              # 流程级工具
    primary_agent_key: Optional[str] = None   # 主 Agent 的 key

    class Config:
        arbitrary_types_allowed = True  # 允许任意类型（因为 BaseAgent 不是标准 Pydantic 模型）

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        """
        初始化流程，支持多种方式传入 Agent：
          - 单个 Agent：自动包装为 {"default": agent}
          - Agent 列表：自动包装为 {"agent_0": a, "agent_1": b, ...}
          - Agent 字典：直接使用（推荐方式，可以自定义命名）
        """
        # 处理不同的 Agent 提供方式，统一转换为字典
        if isinstance(agents, BaseAgent):
            agents_dict = {"default": agents}                  # 单个 Agent
        elif isinstance(agents, list):
            agents_dict = {f"agent_{i}": agent for i, agent in enumerate(agents)}  # 列表
        else:
            agents_dict = agents                                # 已经是字典

        # 如果未指定主 Agent，默认使用第一个
        primary_key = data.get("primary_agent_key")
        if not primary_key and agents_dict:
            primary_key = next(iter(agents_dict))
            data["primary_agent_key"] = primary_key

        # 设置 agents 字典
        data["agents"] = agents_dict

        # 调用 Pydantic BaseModel 的初始化
        super().__init__(**data)

    @property
    def primary_agent(self) -> Optional[BaseAgent]:
        """获取流程的主 Agent（通过 primary_agent_key 从字典中查找）"""
        return self.agents.get(self.primary_agent_key)

    def get_agent(self, key: str) -> Optional[BaseAgent]:
        """通过 key 获取特定的 Agent"""
        return self.agents.get(key)

    def add_agent(self, key: str, agent: BaseAgent) -> None:
        """向流程添加新的 Agent（可以动态添加）"""
        self.agents[key] = agent

    @abstractmethod
    async def execute(self, input_text: str) -> str:
        """
        执行流程的抽象方法，由子类实现。
        例如 PlanningFlow 会先规划任务再逐步执行。
        """
