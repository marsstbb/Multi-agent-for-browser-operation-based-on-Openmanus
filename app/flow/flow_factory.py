"""
flow/flow_factory.py —— 流程工厂

这是一个工厂模式（Factory Pattern）的实现：
  - FlowType: 流程类型枚举（目前只有 PLANNING）
  - FlowFactory: 根据类型创建对应的流程实例

使用示例：
    flow = FlowFactory.create_flow(FlowType.PLANNING, agents)
    result = await flow.execute("帮我完成一项任务")

工厂模式的好处：
  - 调用者不需要知道具体创建哪个 Flow 类
  - 添加新流程类型时只需在字典中注册，不需要修改调用方
"""

from enum import Enum
from typing import Dict, List, Union

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.flow.planning import PlanningFlow


class FlowType(str, Enum):
    """
    流程类型枚举。
    目前只支持 PLANNING（规划流程），但架构上支持扩展更多类型。
    例如未来可以添加：
      - DEBATE: 多 Agent 辩论流程
      - PIPELINE: Agent 流水线流程
    """
    PLANNING = "planning"  # 规划流程：先拆分任务再逐步执行


class FlowFactory:
    """
    流程工厂 —— 根据类型创建对应的流程实例。
    这是一个静态方法类（所有方法都是 @staticmethod），不需要实例化。
    """

    @staticmethod
    def create_flow(
        flow_type: FlowType,
        agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]],
        **kwargs,
    ) -> BaseFlow:
        """
        创建指定类型的流程。

        Args:
            flow_type: 流程类型（如 FlowType.PLANNING）
            agents: Agent 实例（支持单个、列表或字典）
            **kwargs: 其他参数（传递给具体的 Flow 构造函数）

        Returns:
            创建的流程实例

        Raises:
            ValueError: 如果流程类型未知
        """
        # 流程类型到具体类的映射表
        flows = {
            FlowType.PLANNING: PlanningFlow,
        }

        # 查找对应的流程类
        flow_class = flows.get(flow_type)
        if not flow_class:
            raise ValueError(f"Unknown flow type: {flow_type}")

        # 实例化并返回
        return flow_class(agents, **kwargs)
