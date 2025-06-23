"""
基于LangGraph的SCID-5问诊代理 - 模块化版本
实现结构化的精神疾病问诊流程

这个文件现在作为主要的导入入口，保持向后兼容性。
实际的功能已经被分拆到多个专门的模块中：

- types.py: 类型定义
- agent_core.py: 核心代理类
- conversation_modes.py: 对话模式处理
- interview_flow.py: 问诊流程
- response_generation.py: 回应生成
- emergency_handling.py: 紧急情况处理
- workflow_builder.py: 工作流程构建
- scid5_agent.py: 主要代理类
- factory.py: 工厂函数和全局实例
"""

# 导入所有类型定义
from agent_types import InterviewState

# 导入主要的代理类
from scid5_agent import SCID5Agent

# 导入工厂函数和全局实例
from factory import create_agent, scid5_agent

# 导入各个功能模块（供高级用户使用）
from agent_core import SCID5AgentCore
from conversation_modes import ConversationModeHandler  
from interview_flow import InterviewFlowHandler
from response_generation import ResponseGenerator
from emergency_handling import EmergencyHandler
from workflow_builder import WorkflowBuilder

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

# 注意：原来的大型 SCID5Agent 类实现已经被分拆到以下模块中：
# - scid5_agent.py: 主要代理类实现
# - agent_core.py: 核心功能
# - conversation_modes.py: 对话模式处理
# - interview_flow.py: 问诊流程
# - response_generation.py: 回应生成
# - emergency_handling.py: 紧急情况处理
# - workflow_builder.py: 工作流构建
#
# 原来的类现在通过导入重新导出，保持向后兼容性