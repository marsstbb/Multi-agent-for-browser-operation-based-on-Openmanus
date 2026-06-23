"""
main.py —— 项目的命令行入口

这是最简单的启动方式，通过命令行运行：
    python main.py --prompt "帮我搜索今天的新闻"
或者：
    python main.py    # 然后在提示中输入你的需求

运行流程：
    1. 解析命令行参数（--prompt）
    2. 创建 Manus Agent（自动连接 MCP 服务器）
    3. 运行 agent.run(prompt)，开始 think→act 循环
    4. 完成后清理资源（关闭浏览器、断开 MCP 等）

对比其他入口：
    - main.py    —— 命令行，单 Agent，最简单
    - app.py     —— Web UI，单 Agent，支持浏览器访问
    - run_flow.py —— 命令行，多 Agent 协作，可以分解任务
"""

import argparse   # 命令行参数解析库
import asyncio    # 异步 I/O 框架（因为 Agent 的 think/act 都是异步的）

from app.agent.manus import Manus   # 核心通用 Agent
from app.logger import logger       # 日志工具


async def main():
    # ========== 第 1 步：解析命令行参数 ==========
    # argparse 是 Python 的命令行参数解析库
    # 例如：python main.py --prompt "帮我写一个 Python 程序"
    # 这里的 --prompt 就是可选参数，如果不提供，会在下面提示用户输入
    parser = argparse.ArgumentParser(description="运行 Manus agent")
    parser.add_argument(
        "--prompt", type=str, required=False, help="输入给 agent 的提示"
    )
    args = parser.parse_args()

    # ========== 第 2 步：创建 Manus Agent ==========
    # Manus.create() 是异步工厂方法，会：
    #   1. 创建 Manus 实例（包含所有内置工具）
    #   2. 连接配置文件中指定的 MCP 服务器
    # 这里用 await 是因为创建过程中有异步操作（网络连接）
    agent = await Manus.create()
    try:
        # ========== 第 3 步：获取用户输入 ==========
        # 如果命令行提供了 --prompt 参数，就用它；否则提示用户输入
        prompt = args.prompt if args.prompt else input("请输入你的提示: ")
        if not prompt.strip():
            logger.warning("提供的提示为空。")
            return

        # ========== 第 4 步：开始运行 Agent ==========
        # agent.run() 会启动 think→act 循环，直到任务完成或达到最大步数
        logger.warning("正在处理你的请求...")
        await agent.run(prompt)
        logger.info("请求处理完成。")
    except KeyboardInterrupt:
        # Ctrl+C 中断时捕获异常，优雅退出
        logger.warning("操作被中断。")
    finally:
        # ========== 第 5 步：清理资源 ==========
        # 无论成功还是失败，都必须清理资源
        # cleanup() 会关闭浏览器、断开 MCP 服务器连接
        await agent.cleanup()


# ========== 程序入口 ==========
# 当直接运行此文件时（python main.py），执行 main() 函数
# asyncio.run() 会创建一个事件循环来运行异步函数
if __name__ == "__main__":
    asyncio.run(main())
