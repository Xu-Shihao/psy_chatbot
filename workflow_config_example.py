"""
SCID5 Agent工作流程模式配置示例

本文件展示了如何配置和使用不同的工作流程模式
"""

from src.langgraph_agent import SCID5Agent, create_agent

# 方式1: 直接创建不同模式的代理实例

# 智能检测模式 - 根据用户意图自由切换问诊和闲聊
adaptive_agent = SCID5Agent(workflow_mode="adaptive")

# 固定流程模式 - 先完成问诊再转CBT闲聊
structured_agent = SCID5Agent(workflow_mode="structured")

# 方式2: 使用工厂函数创建

# 创建智能检测模式的代理
agent_adaptive = create_agent("adaptive")

# 创建固定流程模式的代理
agent_structured = create_agent("structured")

# 使用示例
def example_usage():
    """工作流程模式使用示例"""
    
    print("=== 智能检测模式示例 ===")
    print("特点：用户可以自由选择问诊或闲聊，系统会自动检测意图")
    
    # 使用智能检测模式
    agent = create_agent("adaptive")
    
    # 用户可以说："我想聊聊天" -> 进入CBT闲聊模式
    # 用户也可以说："我最近很焦虑" -> 进入问诊模式
    # 系统会自动检测并切换
    
    print("\n=== 固定流程模式示例 ===")
    print("特点：强制先完成问诊评估，然后自动转到CBT闲聊")
    
    # 使用固定流程模式
    agent = create_agent("structured")
    
    # 无论用户说什么，都会先进行完整的问诊评估
    # 完成问诊后，自动转到CBT闲聊模式
    # 适合需要完整评估流程的场景

def process_conversation_example():
    """对话处理示例"""
    
    # 创建固定流程模式的代理
    agent = create_agent("structured")
    
    # 开始对话
    response, state = agent.process_message_sync("开始评估")
    print("AI:", response)
    
    # 用户回答（在固定模式下，会强制进入问诊流程）
    response, state = agent.process_message_sync("我最近感觉很累", state)
    print("AI:", response)
    
    # 继续对话...
    # 当问诊完成后，会自动提示用户可以进行CBT闲聊

if __name__ == "__main__":
    example_usage()
    print("\n" + "="*50)
    process_conversation_example()

# 配置建议：

# 1. 如果你的应用场景需要灵活性，用户可能只想聊天或只想问诊：
#    使用 workflow_mode="adaptive"

# 2. 如果你的应用场景需要确保完整的评估流程：
#    使用 workflow_mode="structured"

# 3. 可以根据用户类型或场景动态选择：
def get_agent_for_user(user_type: str) -> SCID5Agent:
    """根据用户类型返回合适的代理"""
    if user_type == "first_time":
        # 首次用户，使用固定流程确保完整评估
        return create_agent("structured")
    elif user_type == "returning":
        # 回访用户，使用智能检测允许灵活对话
        return create_agent("adaptive")
    else:
        # 默认使用智能检测
        return create_agent("adaptive") 