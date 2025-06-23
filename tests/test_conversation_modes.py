#!/usr/bin/env python
"""
测试对话模式识别功能
"""

import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.langgraph_agent import SCID5Agent

def test_conversation_modes():
    """测试对话模式识别"""
    print("🧪 开始测试对话模式识别功能...")
    
    try:
        # 创建代理实例
        agent = SCID5Agent()
        print("✅ 代理创建成功")
        
        # 测试1: 开始评估
        print("\n📝 测试1: 开始评估")
        response1, state1 = agent.process_message_sync("开始评估")
        print(f"回复: {response1[:100]}...")
        print(f"模式: {state1.get('conversation_mode', 'N/A')}")
        
        # 测试2: 想要闲聊
        print("\n📝 测试2: 用户想要闲聊")
        response2, state2 = agent.process_message_sync("我想和你闲聊一下", state1)
        print(f"回复: {response2[:100]}...")
        print(f"模式: {state2.get('conversation_mode', 'N/A')}")
        print(f"CBT疗愈师激活: {state2.get('chat_therapist_active', False)}")
        
        # 测试3: 提到心理问题
        print("\n📝 测试3: 用户提到心理问题")
        response3, state3 = agent.process_message_sync("我最近感觉很抑郁，心情很不好", state1)
        print(f"回复: {response3[:100]}...")
        print(f"模式: {state3.get('conversation_mode', 'N/A')}")
        
        # 测试4: 模式检测分析
        print("\n📝 测试4: 检查模式检测结果")
        detection_result = state2.get('mode_detection_result', {})
        if detection_result:
            print(f"检测模式: {detection_result.get('detected_mode', 'N/A')}")
            print(f"置信度: {detection_result.get('confidence', 0)}")
            print(f"理由: {detection_result.get('reason', 'N/A')}")
        
        print("\n✅ 所有测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_conversation_modes()
    exit(0 if success else 1) 