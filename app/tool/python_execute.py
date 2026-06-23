"""
tool/python_execute.py - Python 代码执行工具
==============================================
允许 Agent 执行 Python 代码。

安全设计：
- 使用 multiprocessing.Process 在子进程中执行代码，与主进程隔离
- 设置超时时间（默认 5 秒），防止死循环卡住 Agent
- 捕获 stdout 输出作为返回结果

为什么用多进程而不是 exec()？
直接在主进程执行用户代码太危险，可能：
- 死循环导致主进程卡死
- 恶意代码影响主进程数据
多进程可以通过 terminate() 强制终止。
"""

import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """Python 代码执行工具

    当 Agent 需要执行计算、数据处理、测试代码时，会使用这个工具。
    Agent 会生成 Python 代码，然后由这个工具在隔离环境中执行。
    """

    name: str = "python_execute"
    description: str = "执行 Python 代码字符串。注意：只有 print 输出可见，函数返回值不会被捕获。使用 print 语句查看结果。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码。",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        """在子进程中执行代码（内部方法）

        为什么重定向 stdout？
        exec() 不会返回值，代码的输出通过 print() 实现。
        将 stdout 重定向到 StringIO，就可以捕获 print 的输出。
        """
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        使用超时执行提供的 Python 代码。

        Args:
            code (str): 要执行的 Python 代码。
            timeout (int): 执行超时时间（秒）。

        Returns:
            Dict: 包含执行输出或错误消息的 'output' 和 'success' 状态。
        """

        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            # 创建子进程执行代码（与主进程隔离）
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)  # 等待子进程完成，最多等待 timeout 秒

            # 如果子进程还在运行，说明超时了，强制终止
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"Execution timeout after {timeout} seconds",
                    "success": False,
                }
            return dict(result)
