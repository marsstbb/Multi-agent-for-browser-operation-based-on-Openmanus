"""
tool/ask_human.py —— 人工交互工具

当 Agent 遇到无法自己解决的问题时，可以通过这个工具向人类提问。

工作机制：
  1. LLM 决定需要人类帮助，调用 ask_human 工具
  2. 工具在控制台打印问题，并等待用户输入
  3. 用户的回答会被作为工具执行结果返回给 LLM
  4. LLM 根据回答继续执行

使用场景：
  - 当需要用户提供敏感信息（如密码、账号）时
  - 当 LLM 不确定用户的意图，需要确认时
  - 注意：在 Web UI 模式下这个工具无法使用（只能在命令行中使用）
"""

from app.tool import BaseTool


class AskHuman(BaseTool):
    """人工交互工具 —— 向人类提问并等待回答"""

    name: str = "ask_human"  # 工具名称
    description: str = "使用此工具向人类寻求帮助。"  # LLM 看到的工具描述
    # 参数定义：只有一个 inquire 字段，表示要问人类的问题
    parameters: str = {
        "type": "object",
        "properties": {
            "inquire": {
                "type": "string",
                "description": "你想问人类的问题。",
            }
        },
        "required": ["inquire"],
    }

    async def execute(self, inquire: str) -> str:
        """
        执行工具：在控制台显示问题并等待用户输入。
        注意：这只在命令行模式下有效，Web UI 模式下无法使用 input()
        """
        return input(f"""Bot: {inquire}\n\nYou: """).strip()
