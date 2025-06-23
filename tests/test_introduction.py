#!/usr/bin/env python
"""
测试自我介绍和开场白功能
"""

import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from src.langgraph_agent import SCID5Agent

def test_introduction():
    """测试自我介绍功能"""
    print("🧪 开始测试自我介绍功能...")
    
    try:
        # 创建代理实例
        agent = SCID5Agent()
        print("✅ 代理创建成功")
        
        # 测试开始评估时的自我介绍
        print("\n📝 测试开始评估时的自我介绍")
        response, state = agent.process_message_sync("开始评估")
        
        print("=" * 60)
        print("AI的自我介绍和开场白:")
        print(response)
        print("=" * 60)
        
        # 验证关键元素是否存在
        print("\n🔍 验证自我介绍内容:")
        
        checks = [
            ("包含名字", "灵犀智伴" in response),
            ("表明身份", "心理咨询师" in response or "咨询师" in response),
            ("表达欢迎", "欢迎" in response or "很高兴" in response or "相遇" in response),
            ("引导分享", "分享" in response or "告诉我" in response or "聊聊" in response),
            ("询问主诉", "困扰" in response or "问题" in response or "感受" in response),
            ("包含重要说明", "重要说明" in response),
            ("保密承诺", "保密" in response),
            ("专业声明", "不能替代" in response and "专业" in response)
        ]
        
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"{status} {check_name}: {check_result}")
        
        # 检查状态
        print(f"\n📊 状态信息:")
        print(f"对话模式: {state.get('conversation_mode')}")
        print(f"当前问题ID: {state.get('current_question_id')}")
        print(f"消息数量: {len(state.get('messages', []))}")
        
        # 验证是否成功
        all_passed = all(check[1] for check in checks)
        if all_passed:
            print("\n✅ 自我介绍测试通过！")
        else:
            print("\n❌ 部分检查未通过")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_introduction()
    exit(0 if success else 1) 