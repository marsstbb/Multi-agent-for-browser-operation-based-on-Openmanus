"""
agent/toolcall.py - 工具调用 Agent
====================================
这是 ReActAgent 的具体实现，实现了 LLM Function Calling 机制。

这是整个项目中最核心的文件之一，它定义了 Agent 如何：
1. think(): 调用 LLM，让 LLM 决定调用哪些工具
2. act(): 执行 LLM 决定的工具调用
3. execute_tool(): 执行单个工具调用
4. _handle_special_tool(): 处理特殊工具（如 Terminate 结束 Agent）

调用链路：
  BaseAgent.run()
    → step() = think() + act()
      → think(): llm.ask_tool(tools=...) → LLM 返回 tool_calls
      → act(): 遍历 tool_calls → execute_tool() → 结果添加到 memory
"""

import asyncio
import json
from typing import Any, List, Optional, Union

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "需要工具调用但未提供"


class ToolCallAgent(ReActAgent):
    """工具调用 Agent

    实现了 LLM Function Calling 的具体逻辑：
    - think(): 将可用工具列表发给 LLM，LLM 返回要调用的工具列表
    - act(): 遍历 LLM 返回的 tool_calls，逐个执行

    这是所有具体 Agent（Manus、BrowserAgent、SWEAgent 等）的直接父类。
    """

    name: str = "toolcall"
    description: str = "可以执行工具调用的 agent。"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # ---- 工具配置 ----
    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()  # 默认工具：对话完成 + 终止
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO  # type: ignore  # auto=LLM自己决定是否用工具
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])  # 特殊工具（会改变 Agent 状态）

    # ---- 运行时状态 ----
    tool_calls: List[ToolCall] = Field(default_factory=list)  # 当前步骤 LLM 请求的工具调用列表
    _current_base64_image: Optional[str] = None  # 当前工具返回的图片（如浏览器截图）

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """思考阶段：调用 LLM 决定下一步行动

        核心流程：
        1. 将对话历史 + 可用工具列表发给 LLM
        2. LLM 返回：文本内容 + tool_calls 列表
        3. 将 LLM 的回复添加到内存
        4. 返回 True（有工具调用或内容）或 False（无需行动）
        """
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # 获取带有工具选项的响应
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # 检查这是否是包含 TokenLimitExceeded 的 RetryError
            if hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):
                token_limit_error = e.__cause__
                logger.error(
                    f"🚨 Token limit error (from RetryError): {token_limit_error}"
                )
                self.memory.add_message(
                    Message.assistant_message(
                        f"达到最大 token 限制，无法继续执行: {str(token_limit_error)}"
                    )
                )
                self.state = AgentState.FINISHED
                return False
            raise

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # 记录响应信息
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} selected {len(tool_calls) if tool_calls else 0} tools to use"
        )
        if tool_calls:
            logger.info(
                f"🧰 Tools being prepared: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 Tool arguments: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("No response received from the LLM")

            # 处理不同的 tool_choices 模式
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # 创建并添加 assistant 消息
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # 将在 act() 中处理

            # 对于 'auto' 模式，如果没有命令但存在内容，则继续处理内容
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"处理过程中遇到错误: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """行动阶段：执行 think() 中决定的工具调用

        核心流程：
        1. 遍历 self.tool_calls（由 think() 填充）
        2. 对每个 tool_call 调用 execute_tool()
        3. 将工具返回的结果作为 ToolMessage 添加到内存
        4. 返回所有工具执行结果的拼接文本
        """
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # 如果没有工具调用，返回最后一条消息的内容
            return self.messages[-1].content or "No content or commands to execute"

        results = []
        for command in self.tool_calls:
            # 为每个工具调用重置 base64_image
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 Tool '{command.function.name}' completed its mission! Result: {result}"
            )

            # 将工具响应添加到内存
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """执行单个工具调用

        流程：
        1. 验证工具名称是否合法
        2. 解析 LLM 生成的 JSON 参数
        3. 调用 ToolCollection.execute(name, args) 执行工具
        4. 检查是否是特殊工具（如 Terminate）
        5. 处理工具返回的图片（base64_image）
        6. 返回格式化的观察结果
        """
        if not command or not command.function or not command.function.name:
            return "Error: Invalid command format"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"Error: Unknown tool '{name}'"

        try:
            # 解析参数
            args = json.loads(command.function.arguments or "{}")

            # 执行工具
            logger.info(f"🔧 Activating tool: '{name}'...")
            logger.debug(f"🔧 Tool arguments: {args}")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 处理特殊工具
            await self._handle_special_tool(name=name, result=result)

            # 检查结果是否是带有 base64_image 的 ToolResult
            if hasattr(result, "base64_image") and result.base64_image:
                image_size_kb = len(result.base64_image) * 3 / 4 / 1024
                logger.info(f"📷 Tool '{name}' returned screenshot: {image_size_kb:.2f} KB")
                # 存储 base64_image 以便稍后在 tool_message 中使用
                self._current_base64_image = result.base64_image
            else:
                logger.debug(f"📷 Tool '{name}' did not return screenshot")

            # 调试信息：显示工具执行结果
            if hasattr(result, "error") and result.error:
                logger.error(f"❌ Tool '{name}' failed: {result.error}")
            else:
                result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                logger.debug(f"✅ Tool '{name}' result preview: {result_preview}")

            # 格式化结果以供显示（标准情况）
            observation = (
                f"Observed output of cmd `{name}` executed:\n{str(result)}"
                if result
                else f"Cmd `{name}` completed with no output"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            logger.error(
                f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{command.function.arguments}"
            )
            return f"Error: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具（会改变 Agent 状态的工具）

        特殊工具是指执行后需要改变 Agent 状态的工具，例如：
        - Terminate: 将状态设为 FINISHED，结束 Agent 循环
        """
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # 将 agent 状态设置为已完成
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """确定工具执行是否应该结束 agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """检查工具名称是否在特殊工具列表中"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """清理资源

        在 Agent 执行完成后调用，清理工具使用的资源。
        例如：关闭浏览器、断开 MCP 服务器连接等。
        """
        logger.info(f"🧹 Cleaning up resources for agent '{self.name}'...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and asyncio.iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 Cleaning up tool: {tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 Error cleaning up tool '{tool_name}': {e}", exc_info=True
                    )
        logger.info(f"✨ Cleanup complete for agent '{self.name}'.")

    async def run(self, request: Optional[str] = None) -> str:
        """运行 agent，完成后进行清理。"""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
