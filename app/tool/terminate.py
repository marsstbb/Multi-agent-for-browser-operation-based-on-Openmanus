"""
tool/terminate.py —— 终止工具

这是一个特殊的工具，当 Agent 完成任务时调用它来结束运行。

工作机制：
  - Terminate 被注册为 special_tool_names（在 Manus、BrowserAgent 等中）
  - 当 ToolCallAgent.act() 检测到 LLM 调用了 Terminate 时，
    会将 Agent 状态设为 FINISHED，从而退出 run() 的主循环
  - 这就像是一个“结束信号”，告诉系统：“我做完了”

类比：
  就像在循环中调用 break 语句，但它是通过 LLM 自主决定的。
"""

from app.tool.base import BaseTool


_TERMINATE_DESCRIPTION = """当请求已满足或助手无法继续执行任务时终止交互。
当你完成所有任务后，调用此工具来结束工作。"""


class Terminate(BaseTool):
    """终止工具 —— 当任务完成或无法继续时调用，结束 Agent 的运行循环"""
    name: str = "terminate"  # 工具名称
    description: str = _TERMINATE_DESCRIPTION
    # 参数定义：只有一个 status 字段，表示完成状态（success 或 failure）
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "交互的完成状态。",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str) -> str:
        """执行终止，返回完成消息（实际的状态切换在 ToolCallAgent.act() 中处理）"""
        return f"The interaction has been completed with status: {status}"
