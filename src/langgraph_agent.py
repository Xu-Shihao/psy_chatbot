"""
基于LangGraph的SCID-5问诊代理 - 模块化版本
实现结构化的精神疾病问诊流程

这个文件作为导入入口，保持向后兼容性。
功能模块分布：

- agent.py: 主要代理类和核心功能
- agent_types.py: 类型定义
- conversation_modes.py: 对话模式处理
- interview_flow.py: 问诊流程
- response_generation.py: 回应生成
- emergency_handling.py: 紧急情况处理
- workflow_builder.py: 工作流程构建
- factory.py: 工厂函数和全局实例
"""

# 导入所有类型定义
from agent import InterviewState

# 导入主要的代理类
from agent import SCID5Agent

# 导入工厂函数
from factory import create_agent
# 导入全局实例获取函数
from agent import get_scid5_agent

# 导入各个功能模块（供高级用户使用）
from agent import SCID5AgentCore
from conversation_modes import ConversationModeHandler  
from interview_flow import InterviewFlowHandler
from response_generation import ResponseGenerator
from emergency_handling import EmergencyHandler
from workflow_builder import WorkflowBuilder

# 为了向后兼容，创建全局实例
scid5_agent = get_scid5_agent()

# 保持向后兼容性的导出
__all__ = [
    # 主要类和类型
    'InterviewState',
    'SCID5Agent',
    
    # 工厂函数和全局实例
    'create_agent', 
    'scid5_agent',
    
    # 功能模块（可选，供高级用户使用）
    'SCID5AgentCore',
    'ConversationModeHandler',
    'InterviewFlowHandler', 
    'ResponseGenerator',
    'EmergencyHandler',
    'WorkflowBuilder'
]

# 注意：SCID5Agent 类实现统一在 agent.py 中：
# - agent.py: 主要代理类实现和核心功能
# - conversation_modes.py: 对话模式处理
# - interview_flow.py: 问诊流程
# - response_generation.py: 回应生成
# - emergency_handling.py: 紧急情况处理
# - workflow_builder.py: 工作流构建
#
# 所有类通过导入重新导出，保持向后兼容性