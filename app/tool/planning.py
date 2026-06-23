"""
tool/planning.py —— 计划管理工具

这是一个供 LLM 使用的“计划管理工具”，允许 Agent 创建和管理任务计划。
你可以把它想象成一个“待办事项管理器”：
  - 创建计划（包含多个步骤）
  - 更新计划（修改步骤）
  - 查看计划（查看进度）
  - 标记步骤状态（未开始/进行中/已完成/被阻塞）
  - 删除计划

这个工具在 PlanningFlow 中被使用：
  1. LLM 通过 function calling 调用此工具的 "create" 命令创建计划
  2. PlanningFlow 循环调用此工具的 "mark_step" 命令更新步骤状态
  3. PlanningFlow 调用此工具的 "get" 命令获取当前计划进度

数据结构：
  plans = {
      "plan_123": {
          "plan_id": "plan_123",
          "title": "帮我搜索新闻",
          "steps": ["搜索新闻", "分析内容", "写报告"],
          "step_statuses": ["completed", "in_progress", "not_started"],
          "step_notes": ["找到 3 条新闻", "", ""]
      }
  }
"""

from typing import Dict, List, Literal, Optional

from app.exceptions import ToolError        # 工具错误异常
from app.tool.base import BaseTool, ToolResult  # 工具基类和结果封装


_PLANNING_TOOL_DESCRIPTION = """
一个规划工具，允许 agent 创建和管理用于解决复杂任务的计划。
该工具提供创建计划、更新计划步骤和跟踪进度的功能。
"""


class PlanningTool(BaseTool):
    """
    计划管理工具 —— 支持创建、更新、查询、删除计划。

    继承自 BaseTool，是一个可以被 LLM 通过 function calling 调用的工具。
    LLM 可以发送如下指令：
      {"command": "create", "plan_id": "plan_1", "title": "搜索新闻", "steps": [...]}
      {"command": "mark_step", "plan_id": "plan_1", "step_index": 0, "step_status": "completed"}
      {"command": "get", "plan_id": "plan_1"}
    """

    name: str = "planning"  # 工具名称，LLM 通过这个名字调用
    description: str = _PLANNING_TOOL_DESCRIPTION
    # parameters: 工具的参数定义（JSON Schema 格式）
    # LLM 会根据这个定义知道可以传哪些参数
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "要执行的命令。可用命令：create, update, list, get, set_active, mark_step, delete。",
                "enum": [
                    "create",
                    "update",
                    "list",
                    "get",
                    "set_active",
                    "mark_step",
                    "delete",
                ],
                "type": "string",
            },
            "plan_id": {
                "description": "计划的唯一标识符。create、update、set_active 和 delete 命令需要。get 和 mark_step 命令可选（如果未指定则使用活动计划）。",
                "type": "string",
            },
            "title": {
                "description": "计划的标题。create 命令需要，update 命令可选。",
                "type": "string",
            },
            "steps": {
                "description": "计划步骤列表。create 命令需要，update 命令可选。",
                "type": "array",
                "items": {"type": "string"},
            },
            "step_index": {
                "description": "要更新的步骤索引（从 0 开始）。mark_step 命令需要。",
                "type": "integer",
            },
            "step_status": {
                "description": "为步骤设置的状态。与 mark_step 命令一起使用。",
                "enum": ["not_started", "in_progress", "completed", "blocked"],
                "type": "string",
            },
            "step_notes": {
                "description": "步骤的附加注释。mark_step 命令可选。",
                "type": "string",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    plans: dict = {}  # 按 plan_id 存储所有计划的字典（核心数据存储）
    _current_plan_id: Optional[str] = None  # 当前活动计划的 ID

    async def execute(
        self,
        *,
        command: Literal[
            "create", "update", "list", "get", "set_active", "mark_step", "delete"
        ],
        plan_id: Optional[str] = None,
        title: Optional[str] = None,
        steps: Optional[List[str]] = None,
        step_index: Optional[int] = None,
        step_status: Optional[
            Literal["not_started", "in_progress", "completed", "blocked"]
        ] = None,
        step_notes: Optional[str] = None,
        **kwargs,
    ):
        """
        工具执行入口 —— 根据 command 参数分发到不同的处理方法。

        支持的命令：
          - create: 创建新计划
          - update: 更新现有计划
          - list: 列出所有计划
          - get: 获取特定计划的详情
          - set_active: 设置活动计划
          - mark_step: 标记步骤状态
          - delete: 删除计划
        """

        # 根据命令类型分发到对应的处理方法
        if command == "create":
            return self._create_plan(plan_id, title, steps)
        elif command == "update":
            return self._update_plan(plan_id, title, steps)
        elif command == "list":
            return self._list_plans()
        elif command == "get":
            return self._get_plan(plan_id)
        elif command == "set_active":
            return self._set_active_plan(plan_id)
        elif command == "mark_step":
            return self._mark_step(plan_id, step_index, step_status, step_notes)
        elif command == "delete":
            return self._delete_plan(plan_id)
        else:
            raise ToolError(
                f"Unrecognized command: {command}. Allowed commands are: create, update, list, get, set_active, mark_step, delete"
            )

    def _create_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """
        创建新计划。
        验证参数后，将计划存入 self.plans 字典，并设为活动计划。
        """
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: create")

        if plan_id in self.plans:
            raise ToolError(
                f"A plan with ID '{plan_id}' already exists. Use 'update' to modify existing plans."
            )

        if not title:
            raise ToolError("Parameter `title` is required for command: create")

        if (
            not steps
            or not isinstance(steps, list)
            or not all(isinstance(step, str) for step in steps)
        ):
            raise ToolError(
                "Parameter `steps` must be a non-empty list of strings for command: create"
            )

        # 创建计划数据，每个步骤初始状态为 "not_started"
        plan = {
            "plan_id": plan_id,
            "title": title,
            "steps": steps,
            "step_statuses": ["not_started"] * len(steps),
            "step_notes": [""] * len(steps),
        }

        self.plans[plan_id] = plan
        self._current_plan_id = plan_id  # 新计划自动成为活动计划

        return ToolResult(
            output=f"Plan created successfully with ID: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _update_plan(
        self, plan_id: Optional[str], title: Optional[str], steps: Optional[List[str]]
    ) -> ToolResult:
        """
        更新现有计划的标题或步骤。
        更新步骤时会智能保留已完成步骤的状态和注释。
        """
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: update")

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        plan = self.plans[plan_id]

        if title:
            plan["title"] = title

        if steps:
            if not isinstance(steps, list) or not all(
                isinstance(step, str) for step in steps
            ):
                raise ToolError(
                    "Parameter `steps` must be a list of strings for command: update"
                )

            # 智能保留步骤状态：如果新步骤在相同位置与旧步骤相同，保留其状态和注释
            old_steps = plan["steps"]
            old_statuses = plan["step_statuses"]
            old_notes = plan["step_notes"]

            # 创建新的步骤状态和注释
            new_statuses = []
            new_notes = []

            for i, step in enumerate(steps):
                # 如果步骤在旧步骤的相同位置存在，保留状态和注释
                if i < len(old_steps) and step == old_steps[i]:
                    new_statuses.append(old_statuses[i])
                    new_notes.append(old_notes[i])
                else:
                    new_statuses.append("not_started")
                    new_notes.append("")

            plan["steps"] = steps
            plan["step_statuses"] = new_statuses
            plan["step_notes"] = new_notes

        return ToolResult(
            output=f"Plan updated successfully: {plan_id}\n\n{self._format_plan(plan)}"
        )

    def _list_plans(self) -> ToolResult:
        """列出所有可用计划及其进度摘要"""
        if not self.plans:
            return ToolResult(
                output="No plans available. Create a plan with the 'create' command."
            )

        output = "Available plans:\n"
        for plan_id, plan in self.plans.items():
            current_marker = " (active)" if plan_id == self._current_plan_id else ""
            completed = sum(
                1 for status in plan["step_statuses"] if status == "completed"
            )
            total = len(plan["steps"])
            progress = f"{completed}/{total} steps completed"
            output += f"• {plan_id}{current_marker}: {plan['title']} - {progress}\n"

        return ToolResult(output=output)

    def _get_plan(self, plan_id: Optional[str]) -> ToolResult:
        """获取特定计划的详细信息，如果不指定 plan_id 则返回活动计划"""
        if not plan_id:
            # 如果未提供 plan_id，使用当前活动计划
            if not self._current_plan_id:
                raise ToolError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        plan = self.plans[plan_id]
        return ToolResult(output=self._format_plan(plan))

    def _set_active_plan(self, plan_id: Optional[str]) -> ToolResult:
        """将指定计划设置为活动计划"""
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: set_active")

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        self._current_plan_id = plan_id
        return ToolResult(
            output=f"Plan '{plan_id}' is now the active plan.\n\n{self._format_plan(self.plans[plan_id])}"
        )

    def _mark_step(
        self,
        plan_id: Optional[str],
        step_index: Optional[int],
        step_status: Optional[str],
        step_notes: Optional[str],
    ) -> ToolResult:
        """
        标记指定步骤的状态（如"已完成"、"进行中"等）。
        这是 PlanningFlow 中最常调用的方法。
        """
        if not plan_id:
            # 如果未提供 plan_id，使用当前活动计划
            if not self._current_plan_id:
                raise ToolError(
                    "No active plan. Please specify a plan_id or set an active plan."
                )
            plan_id = self._current_plan_id

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        if step_index is None:
            raise ToolError("Parameter `step_index` is required for command: mark_step")

        plan = self.plans[plan_id]

        if step_index < 0 or step_index >= len(plan["steps"]):
            raise ToolError(
                f"Invalid step_index: {step_index}. Valid indices range from 0 to {len(plan['steps'])-1}."
            )

        if step_status and step_status not in [
            "not_started",
            "in_progress",
            "completed",
            "blocked",
        ]:
            raise ToolError(
                f"Invalid step_status: {step_status}. Valid statuses are: not_started, in_progress, completed, blocked"
            )

        if step_status:
            plan["step_statuses"][step_index] = step_status

        if step_notes:
            plan["step_notes"][step_index] = step_notes

        return ToolResult(
            output=f"Step {step_index} updated in plan '{plan_id}'.\n\n{self._format_plan(plan)}"
        )

    def _delete_plan(self, plan_id: Optional[str]) -> ToolResult:
        """删除指定计划，如果是活动计划则清除活动标记"""
        if not plan_id:
            raise ToolError("Parameter `plan_id` is required for command: delete")

        if plan_id not in self.plans:
            raise ToolError(f"No plan found with ID: {plan_id}")

        del self.plans[plan_id]

        # 如果删除的是活动计划，清除活动标记
        if self._current_plan_id == plan_id:
            self._current_plan_id = None

        return ToolResult(output=f"Plan '{plan_id}' has been deleted.")

    def _format_plan(self, plan: Dict) -> str:
        """
        格式化计划为可读的文本输出。
        输出示例：
          Plan: 搜索新闻 (ID: plan_123)
          ==========================
          Progress: 2/3 steps completed (66.7%)
          Status: 2 completed, 1 in progress, 0 blocked, 0 not started
          Steps:
          0. [✓] 搜索今天的新闻
          1. [✓] 获取新闻详情
          2. [→] 写报告
        """
        # 计算进度统计
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        # 构建输出文本
        output = f"Plan: {plan['title']} (ID: {plan['plan_id']})\n"
        output += "=" * len(output) + "\n\n"

        # 计算进度统计
        total_steps = len(plan["steps"])
        completed = sum(1 for status in plan["step_statuses"] if status == "completed")
        in_progress = sum(
            1 for status in plan["step_statuses"] if status == "in_progress"
        )
        blocked = sum(1 for status in plan["step_statuses"] if status == "blocked")
        not_started = sum(
            1 for status in plan["step_statuses"] if status == "not_started"
        )

        output += f"Progress: {completed}/{total_steps} steps completed "
        if total_steps > 0:
            percentage = (completed / total_steps) * 100
            output += f"({percentage:.1f}%)\n"
        else:
            output += "(0%)\n"

        output += f"Status: {completed} completed, {in_progress} in progress, {blocked} blocked, {not_started} not started\n\n"
        output += "Steps:\n"

        # 添加每个步骤及其状态标记和注释
        for i, (step, status, notes) in enumerate(
            zip(plan["steps"], plan["step_statuses"], plan["step_notes"])
        ):
            # 状态符号：[ ] 未开始、[→] 进行中、[✓] 已完成、[!] 阻塞
            status_symbol = {
                "not_started": "[ ]",
                "in_progress": "[→]",
                "completed": "[✓]",
                "blocked": "[!]",
            }.get(status, "[ ]")

            output += f"{i}. {status_symbol} {step}\n"
            if notes:
                output += f"   Notes: {notes}\n"

        return output
