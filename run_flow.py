"""
run_flow.py —— 多 Agent 协作的入口

当你需要多个 Agent 协作完成复杂任务时，使用这个文件启动。
与 main.py（单 Agent）的区别：
  - main.py：只有一个 Manus Agent，从头做到尾
  - run_flow.py：先让一个“规划 Agent”把任务拆成多个步骤，
           然后由不同的 Agent 分别执行每个步骤

运行方式：
    python run_flow.py
    # 然后在提示中输入你的需求

运行流程：
    1. 创建 Agent 字典（包含 Manus 和可选的 DataAnalysis）
    2. 通过 FlowFactory 创建 PlanningFlow（规划流程）
    3. flow.execute() 会：
       a. 让 LLM 将任务分解为多个步骤（创建计划）
       b. 循环执行每个步骤（选择合适的 Agent）
       c. 所有步骤完成后，生成总结
"""

import asyncio   # 异步 I/O 框架
import time      # 计时器（用于统计执行时间）

from app.agent.data_analysis import DataAnalysis  # 数据分析 Agent
from app.agent.manus import Manus                 # 通用 Agent
from app.config import config                     # 全局配置
from app.flow.flow_factory import FlowFactory, FlowType  # 流程工厂
from app.logger import logger                     # 日志工具


async def run_flow():
    # ========== 第 1 步：创建 Agent 字典 ==========
    # 这里定义了哪些 Agent 可以参与协作：
    #   - "manus": 通用 Agent，能处理大部分任务
    #   - "data_analysis": 数据分析 Agent，专注于数据处理和可视化
    # 字典的 key（如 "manus"）可以在计划步骤中用作标签，例如 "[SEARCH] 搜索新闻"
    agents = {
        "manus": Manus(),
    }
    # 如果配置文件中启用了数据分析 Agent，则加入
    if config.run_flow_config.use_data_analysis_agent:
        agents["data_analysis"] = DataAnalysis()
    try:
        # ========== 第 2 步：获取用户输入 ==========
        prompt = input("Enter your prompt: ")

        if prompt.strip().isspace() or not prompt:
            logger.warning("Empty prompt provided.")
            return

        # ========== 第 3 步：创建规划流程 ==========
        # FlowFactory.create_flow() 是一个工厂方法：
        #   - flow_type=FlowType.PLANNING 表示创建 PlanningFlow（规划流程）
        #   - agents 是参与协作的 Agent 字典
        # PlanningFlow 的工作模式：先规划（拆分步骤）→再逐步执行
        flow = FlowFactory.create_flow(
            flow_type=FlowType.PLANNING,
            agents=agents,
        )
        logger.warning("Processing your request...")

        try:
            # ========== 第 4 步：执行流程 ==========
            start_time = time.time()  # 记录开始时间
            # flow.execute() 会：
            #   1. 让 LLM 根据用户输入创建计划（拆分步骤）
            #   2. 循环执行每个步骤（选择合适的 Agent）
            #   3. 所有步骤完成后，生成总结
            # asyncio.wait_for() 设置 60 分钟超时，防止任务无限运行
            result = await asyncio.wait_for(
                flow.execute(prompt),
                timeout=3600,  # 60 分钟超时
            )
            elapsed_time = time.time() - start_time  # 计算执行时间
            logger.info(f"Request processed in {elapsed_time:.2f} seconds")
            logger.info(result)
        except asyncio.TimeoutError:
            # 超时处理：如果任务太复杂，超过 1 小时还没完成，就强制停止
            logger.error("Request processing timed out after 1 hour")
            logger.info(
                "Operation terminated due to timeout. Please try a simpler request."
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error: {str(e)}")


# ========== 程序入口 ==========
# 当直接运行此文件时（python run_flow.py），启动多 Agent 协作流程
if __name__ == "__main__":
    asyncio.run(run_flow())
