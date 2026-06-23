"""
prompt/toolcall.py —— ToolCallAgent 的基础提示词

这是 ToolCallAgent 的默认提示词，比 Manus 的提示词简单得多。
它只告诉 LLM 两件事：
  1. 你是一个可以执行工具调用的代理
  2. 如果你想停止，就调用 terminate 工具

当 Manus 继承 ToolCallAgent 时，会用自己更详细的提示词覆盖这些默认值。
所以这个文件主要作为“基类默认配置”存在。
"""

# 系统提示词：告诉 LLM 它是一个可以调用工具的 Agent
SYSTEM_PROMPT = "你是一个可以执行工具调用的代理"

# 下一步行动提示词：告诉 LLM 如何结束任务
NEXT_STEP_PROMPT = (
    "如果你想停止交互，请使用 `terminate` 工具/函数调用。"
)
