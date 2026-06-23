"""
flow/planning.py —— PlanningFlow 规划流程

这是多 Agent 协作的核心实现！它实现了"规划-执行"模式：

  第 1 步：创建计划
    用户输入需求 → LLM 将任务拆分成多个步骤
    例如："帮我搜索今天的新闻并写一份报告"
    计划：
      0. [SEARCH] 搜索今天的热点新闻
      1. [SEARCH] 获取新闻详情
      2. [CODE] 生成报告

  第 2 步：循环执行每个步骤
    对于每个步骤：
      - 根据步骤标签（如 [SEARCH]）选择合适的 Agent
      - 调用 agent.run() 执行该步骤
      - 标记步骤为已完成

  第 3 步：生成总结
    所有步骤完成后，让 LLM 生成一份总结报告

继承关系：
  BaseFlow → PlanningFlow
"""

import json       # JSON 处理
import time       # 时间戳（用于生成唯一的计划 ID）
from enum import Enum  # 枚举类
from typing import Dict, List, Optional, Union

from pydantic import Field  # Pydantic 字段定义

from app.agent.base import BaseAgent      # Agent 基类
from app.flow.base import BaseFlow         # 流程基类
from app.llm import LLM                    # 大语言模型
from app.logger import logger              # 日志
from app.schema import AgentState, Message, ToolChoice  # 数据结构
from app.tool import PlanningTool          # 计划管理工具


class PlanStepStatus(str, Enum):
    """
    计划步骤的状态枚举。
    每个步骤都有四种可能的状态：
      - NOT_STARTED: 未开始（等待执行）
      - IN_PROGRESS: 进行中（正在执行）
      - COMPLETED: 已完成（执行成功）
      - BLOCKED: 被阻塞（因为依赖关系无法执行）
    """

    NOT_STARTED = "not_started"    # 未开始
    IN_PROGRESS = "in_progress"    # 进行中
    COMPLETED = "completed"        # 已完成
    BLOCKED = "blocked"            # 被阻塞

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """返回所有可能的步骤状态值列表"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """返回"活动状态"列表（未开始或进行中的步骤都需要继续执行）"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """
        返回状态到标记符号的映射，用于显示：
          [✓] 已完成  [→] 进行中  [!] 被阻塞  [ ] 未开始
        """
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """
    规划流程 —— 多 Agent 协作的核心实现。

    工作流程：
      1. _create_initial_plan(): 让 LLM 创建计划
      2. 主循环：
         a. _get_current_step_info(): 找到第一个未完成的步骤
         b. get_executor(): 选择合适的 Agent
         c. _execute_step(): 执行该步骤
         d. _mark_step_completed(): 标记步骤完成
      3. _finalize_plan(): 所有步骤完成后生成总结

    属性：
      - llm: 流程专用的 LLM，用于规划和总结（不是 Agent 的 LLM）
      - planning_tool: 计划管理工具（创建/更新/查询计划）
      - executor_keys: 可执行任务的 Agent 名称列表
      - active_plan_id: 当前活动计划的唯一 ID
      - current_step_index: 当前正在执行的步骤索引
    """

    llm: LLM = Field(default_factory=lambda: LLM())  # 流程专用 LLM
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)  # 计划管理工具
    executor_keys: List[str] = Field(default_factory=list)  # 执行器 Agent 的 key 列表
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")  # 当前计划 ID
    current_step_index: Optional[int] = None  # 当前执行的步骤索引

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        """
        初始化 PlanningFlow。
        处理特殊参数：
          - executors: 指定哪些 Agent 可以执行任务
          - plan_id: 指定计划 ID（如果不指定，自动生成）
          - planning_tool: 提供自定义的规划工具
        """
        # 处理执行器参数
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 处理计划 ID 参数
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 如果没有提供规划工具，创建一个默认的
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 调用父类 BaseFlow 的初始化
        super().__init__(agents, **data)

        # 如果没有指定执行器，默认所有 Agent 都可以执行
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        根据步骤类型选择合适的执行器 Agent。

        选择逻辑：
          1. 如果步骤有类型标签（如 "search"、"code"），
             且该标签匹配某个 Agent 的 key，则使用该 Agent
          2. 否则使用第一个可用的执行器 Agent
          3. 最后回退到主 Agent

        例如：步骤文本是 "[data_analysis] 分析数据"，
        提取出 step_type="data_analysis"，就返回 DataAnalysis Agent
        """
        # 如果步骤类型匹配某个 Agent 的 key，直接使用该 Agent
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 否则使用第一个可用的执行器
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 回退到主 Agent
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """
        执行规划流程的主方法。

        流程：
          1. 创建初始计划（让 LLM 拆分任务）
          2. 主循环：找到未完成步骤 → 选择 Agent → 执行 → 标记完成
          3. 如果没有更多步骤或 Agent 请求终止，则退出循环
          4. 生成计划总结
        """
        try:
            if not self.primary_agent:
                raise ValueError("No primary agent available")

            # ===== 第 1 阶段：创建初始计划 =====
            # 如果用户提供了输入，让 LLM 创建计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 验证计划是否创建成功
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"Plan creation failed. Plan ID {self.active_plan_id} not found in planning tool."
                    )
                    return f"Failed to create plan for: {input_text}"

            # ===== 第 2 阶段：循环执行每个步骤 =====
            result = ""
            while True:
                # 找到第一个未完成的步骤
                self.current_step_index, step_info = await self._get_current_step_info()

                # 如果没有更多未完成步骤，退出循环
                if self.current_step_index is None:
                    result += await self._finalize_plan()  # 生成总结
                    break

                # 根据步骤类型选择合适的 Agent，然后执行
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 检查 Agent 是否请求终止
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"Error in PlanningFlow: {str(e)}")
            return f"Execution failed: {str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """
        创建初始计划 —— 让 LLM 将用户任务拆分为多个步骤。

        流程：
          1. 构建系统提示词（告诉 LLM 它是一个规划助手）
          2. 如果有多个 Agent，告诉 LLM 每个 Agent 的能力
          3. 调用 LLM + PlanningTool，让 LLM 通过工具调用来创建计划
          4. 如果 LLM 没有成功创建计划，则创建一个默认计划
        """
        logger.info(f"Creating initial plan with ID: {self.active_plan_id}")

        # 构建系统提示词，告诉 LLM 它是一个规划助手
        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )
        # 收集所有可用 Agent 的描述信息
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {
                        "name": key.upper(),
                        "description": self.agents[key].description,
                    }
                )
        # 如果有多个 Agent，告诉 LLM 每个 Agent 的能力
        # 这样 LLM 可以在步骤中指定使用哪个 Agent，例如 [MANUS] 或 [DATA_ANALYSIS]
        if len(agents_description) > 1:
            # 添加 agent 描述以供选择
            system_message_content += (
                f"\nNow we have {agents_description} agents. "
                f"The infomation of them are below: {json.dumps(agents_description)}\n"
                "When creating steps in the planning tool, please specify the agent names using the format '[agent_name]'."
            )

        # 创建系统消息和用户消息
        system_message = Message.system_message(system_message_content)
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        # 调用 LLM，提供 PlanningTool 工具，让 LLM 通过工具调用创建计划
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],  # 将 PlanningTool 转为 LLM 可用的工具格式
            tool_choice=ToolChoice.AUTO,  # 让 LLM 自己决定是否调用工具
        )

        # 处理 LLM 返回的工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析工具调用参数
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)  # 解析 JSON 字符串
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse tool arguments: {args}")
                            continue

                    # 确保 plan_id 正确设置，然后执行工具创建计划
                    args["plan_id"] = self.active_plan_id
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"Plan creation result: {str(result)}")
                    return

        # 如果 LLM 没有成功创建计划，创建一个默认计划（回退机制）
        logger.warning("Creating default plan")

        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],  # 默认三步骤
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        查找第一个未完成的步骤。

        Returns:
            (步骤索引, 步骤信息) 元组。
            如果没有未完成步骤，返回 (None, None)。

        步骤信息示例：
            {"text": "[SEARCH] 搜索新闻", "type": "search"}
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"Plan with ID {self.active_plan_id} not found")
            return None, None

        try:
            # 直接从规划工具中获取计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 遍历所有步骤，找到第一个未完成的
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 找到活动步骤，提取步骤信息
                    step_info = {"text": step}

                    # 尝试从文本中提取步骤类型标签，例如 "[SEARCH]" 或 "[CODE]"
                    # 这个标签可以用来选择合适的 Agent
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 将当前步骤标记为"进行中"
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        # 如果工具调用失败，直接更新状态（回退方案）
                        logger.warning(f"Error marking step as in_progress: {e}")
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # 所有步骤都已完成

        except Exception as e:
            logger.warning(f"Error finding current step index: {e}")
            return None, None  # 出错时也返回 None，让主循环退出

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """
        使用指定的 Agent 执行当前步骤。

        流程：
          1. 获取当前计划状态（已完成的步骤、进度等）
          2. 构建提示词，告诉 Agent 当前计划状态和它需要执行的步骤
          3. 调用 agent.run() 执行
          4. 执行成功后标记步骤为已完成
        """
        # 获取当前计划状态文本（包含所有步骤的进度信息）
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 构建提示词，包含计划上下文和当前任务
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        Please only execute this current step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        # 调用 agent.run() 执行步骤（会启动 Agent 自己的 think→act 循环）
        try:
            step_result = await executor.run(step_prompt)

            # 执行成功后将步骤标记为已完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"Error executing step {self.current_step_index}: {e}")
            return f"Error executing step {self.current_step_index}: {str(e)}"

    async def _mark_step_completed(self) -> None:
        """
        将当前步骤标记为已完成。
        优先通过 PlanningTool 更新，如果失败则直接修改内部数据。
        """
        if self.current_step_index is None:
            return

        try:
            # 优先通过 PlanningTool 更新状态
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"Marked step {self.current_step_index} as completed in plan {self.active_plan_id}"
            )
        except Exception as e:
            # 回退方案：直接修改内部数据
            logger.warning(f"Failed to update plan status: {e}")
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保 step_statuses 列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 直接更新状态为已完成
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """获取当前计划的格式化文本（包含所有步骤的进度信息）"""
        try:
            # 通过 PlanningTool 获取计划文本
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            # 回退方案：直接从内部数据生成
            logger.error(f"Error getting plan: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """回退方法：直接从内部数据生成计划文本（当 PlanningTool 调用失败时使用）"""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"Error: Plan with ID {self.active_plan_id} not found"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "Untitled Plan")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保 step_statuses 和 step_notes 与步骤数量匹配
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 按状态统计步骤数
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            # 计算进度百分比
            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            # 构建计划文本输出
            plan_text = f"Plan: {title} (ID: {self.active_plan_id})\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"Progress: {completed}/{total} steps completed ({progress:.1f}%)\n"
            )
            # 添加各状态的步骤数统计
            plan_text += f"Status: {status_counts[PlanStepStatus.COMPLETED.value]} completed, {status_counts[PlanStepStatus.IN_PROGRESS.value]} in progress, "
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} blocked, {status_counts[PlanStepStatus.NOT_STARTED.value]} not started\n\n"
            plan_text += "Steps:\n"

            # 获取状态标记符号（[✓] [→] [!] [ ]）
            status_marks = PlanStepStatus.get_status_marks()

            # 遍历每个步骤，显示状态、文本和注释
            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # 使用状态标记符号（[✓] 已完成、[→] 进行中、[!] 阻塞、[ ] 未开始）
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   Notes: {notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"Error generating plan text from storage: {e}")
            return f"Error: Unable to retrieve plan with ID {self.active_plan_id}"

    async def _finalize_plan(self) -> str:
        """
        完成计划并生成总结。
        优先使用流程的 LLM 生成总结，如果失败则回退到 Agent 生成。
        """
        plan_text = await self._get_plan_text()

        # 尝试使用流程的 LLM 直接生成总结
        try:
            system_message = Message.system_message(
                "You are a planning assistant. Your task is to summarize the completed plan."
            )

            user_message = Message.user_message(
                f"The plan has been completed. Here is the final plan status:\n\n{plan_text}\n\nPlease provide a summary of what was accomplished and any final thoughts."
            )

            # 调用 LLM 生成总结（这里用 ask 而不是 ask_tool，因为不需要工具调用）
            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"Plan completed:\n\n{response}"
        except Exception as e:
            logger.error(f"Error finalizing plan with LLM: {e}")

            # 回退方案：使用主 Agent 生成总结
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"Error finalizing plan with agent: {e2}")
                return "Plan completed. Error generating summary."
