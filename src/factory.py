"""
工厂函数和全局实例模块
提供创建SCID5Agent实例的工厂函数和全局实例
"""

from scid5_agent import SCID5Agent


def create_agent(workflow_mode: str = "adaptive") -> SCID5Agent:
    """
    创建SCID5代理实例
    
    Args:
        workflow_mode: 工作流程模式
            - "adaptive": 智能检测模式，根据用户意图自由切换问诊和闲聊
            - "structured": 固定流程模式，先完成问诊再转CBT闲聊
    
    Returns:
        SCID5Agent: 配置好的代理实例
    """
    return SCID5Agent(workflow_mode=workflow_mode)


# 全局代理实例
# 可以通过修改这里的workflow_mode参数来切换工作模式：
# - "adaptive": 智能检测模式，根据用户意图自由切换问诊和闲聊
# - "structured": 固定流程模式，先完成问诊再转CBT闲聊
scid5_agent = SCID5Agent(workflow_mode="adaptive")  # 默认使用智能检测模式