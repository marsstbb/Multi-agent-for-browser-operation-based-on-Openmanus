"""
agent/data_analysis.py —— 数据分析 Agent

这是一个专门用于数据分析任务的 Agent，继承自 ToolCallAgent。

与 Manus 的区别：
  - Manus 是全能 Agent，包含浏览器、搜索、文件编辑等多种工具
  - DataAnalysis 专注于数据分析，包含：
    1. NormalPythonExecute: 执行 Python 代码（数据分析专用）
    2. VisualizationPrepare: 准备可视化数据
    3. DataVisualization: 生成图表
    4. Terminate: 终止工具

使用场景：
  在 run_flow.py（多 Agent 协作）中作为专门的"数据分析师"角色，
  当计划步骤包含 [data_analysis] 标签时，会被 PlanningFlow 自动选择。
"""

from pydantic import Field

from app.agent.toolcall import ToolCallAgent  # 父类：工具调用 Agent
from app.config import config                # 全局配置
from app.prompt.visualization import NEXT_STEP_PROMPT, SYSTEM_PROMPT  # 数据分析专用提示词
from app.tool import Terminate, ToolCollection  # 终止工具 + 工具集管理器
# 数据分析专用工具
from app.tool.chart_visualization.chart_prepare import VisualizationPrepare  # 可视化准备
from app.tool.chart_visualization.data_visualization import DataVisualization  # 图表生成
from app.tool.chart_visualization.python_execute import NormalPythonExecute    # Python 执行


class DataAnalysis(ToolCallAgent):
    """
    数据分析 Agent —— 专注于数据处理和可视化的专家。

    在多 Agent 协作（PlanningFlow）中担任"数据分析师"角色，
    当计划步骤包含 [data_analysis] 标签时会被自动选择。

    继承关系：
      BaseAgent → ReActAgent → ToolCallAgent → DataAnalysis
    """

    # Agent 名称和描述（PlanningFlow 根据描述选择 Agent）
    name: str = "Data_Analysis"
    description: str = "一个利用 Python 和数据可视化工具来解决各种数据分析任务的分析 agent"

    # 使用数据分析专用的系统提示词，注入工作目录
    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    # max_observe: 最大观察长度（数据分析结果可能较长，所以比 Manus 的 10000 更大）
    max_observe: int = 15000
    max_steps: int = 20

    # 工具集：数据分析专用工具
    #   1. NormalPythonExecute: 执行 Python 代码（数据分析、统计计算等）
    #   2. VisualizationPrepare: 准备可视化数据（数据清洗、格式转换）
    #   3. DataVisualization: 生成图表（柱状图、折线图、饼图等）
    #   4. Terminate: 终止工具（任务完成时调用）
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            NormalPythonExecute(),
            VisualizationPrepare(),
            DataVisualization(),
            Terminate(),
        )
    )
