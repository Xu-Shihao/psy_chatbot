"""
对话模式处理模块
包含对话模式检测和CBT疗愈师响应功能
"""

import json
from langchain.schema import HumanMessage, SystemMessage

from agent_types import InterviewState
from enhanced_intent_detection import EnhancedIntentDetector


class ConversationModeHandler:
    """对话模式处理器 - 集成增强意图检测"""
    
    def __init__(self, llm, workflow_mode: str = "adaptive"):
        self.llm = llm
        self.workflow_mode = workflow_mode
        
        # 初始化增强意图检测器
        self.intent_detector = EnhancedIntentDetector(llm, workflow_mode)
        
        # 保留原有的简单检测作为后备
        self.simple_detection_enabled = True
    
    def detect_conversation_mode(self, state: InterviewState) -> InterviewState:
        """检测对话模式 - 使用增强检测器"""
        try:
            # 使用增强的意图检测
            result = self.intent_detector.detect_conversation_mode(state)
            
            # 添加调试信息
            if result.get("mode_detection_result"):
                print("🔍 增强意图检测结果：")
                print(f"  模式: {result['conversation_mode']}")
                print(f"  置信度: {result['mode_detection_result'].get('confidence', 'N/A')}")
                print(f"  原因: {result['mode_detection_result'].get('reason', 'N/A')}")
            
            return result
                
        except Exception as e:
            print(f"⚠️ 增强检测失败，使用简单检测: {e}")
            
            # 后备：使用原有的简单检测逻辑
            if self.simple_detection_enabled:
                return self._simple_detection_fallback(state)
            else:
                raise e
    
    def _simple_detection_fallback(self, state: InterviewState) -> InterviewState:
        """简单检测后备逻辑（原有逻辑的简化版）"""
        print("🔄 使用简单检测后备逻辑")
        
        # 如果已经完成评估，默认进入闲聊模式
        if state.get("assessment_complete", False):
            return {
                **state,
                "conversation_mode": "assessment_complete",
                "chat_therapist_active": True,
                "mode_detection_result": {
                    "detected_mode": "assessment_complete",
                    "confidence": 1.0,
                    "reason": "评估已完成，进入后续闲聊模式"
                }
            }
        
        # 获取当前会话轮数
        current_turn = state.get("conversation_turn_count", 0)
        interview_locked = state.get("interview_mode_locked", False)
        
        # 如果问诊模式已锁定，强制继续问诊
        if interview_locked:
            return {
                **state,
                "conversation_mode": "continue_interview",
                "chat_therapist_active": False,
                "conversation_turn_count": current_turn + 1,
                "mode_detection_result": {
                    "detected_mode": "continue_interview",
                    "confidence": 1.0,
                    "reason": "问诊模式已锁定，继续进行问诊直至完成"
                }
            }
        
        # 结构化模式：如果评估未完成，强制进入问诊模式
        if (self.workflow_mode == "structured" and 
            not state.get("assessment_complete", False)):
            return {
                **state,
                "conversation_mode": "continue_interview",
                "chat_therapist_active": False,
                "conversation_turn_count": current_turn + 1,
                "interview_mode_locked": True,
                "mode_detection_result": {
                    "detected_mode": "continue_interview",
                    "confidence": 1.0,
                    "reason": "结构化模式：强制进入问诊流程并锁定"
                }
            }
        
        # 获取最新的用户消息
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return {
                **state,
                "conversation_mode": "interview",
                "conversation_turn_count": current_turn + 1,
                "mode_detection_result": {
                    "detected_mode": "interview",
                    "confidence": 0.8,
                    "reason": "没有用户输入，默认开始问诊"
                }
            }
        
        latest_user_message = user_messages[-1].content
        
        # 简单关键词检测
        chat_keywords = ["闲聊", "聊天", "谈心", "聊聊", "随便聊", "陪我聊"]
        interview_keywords = ["抑郁", "焦虑", "失眠", "情绪", "心理", "症状", "困扰", "问题"]
        
        if any(keyword in latest_user_message for keyword in chat_keywords):
            mode = "chat"
            will_lock = False
        elif any(keyword in latest_user_message for keyword in interview_keywords):
            mode = "interview"
            will_lock = True
        else:
            # 默认继续问诊
            mode = "continue_interview" if state.get("conversation_mode") == "interview" else "interview"
            will_lock = True
        
        return {
            **state,
            "conversation_mode": mode,
            "chat_therapist_active": mode == "chat",
            "conversation_turn_count": current_turn + 1,
            "interview_mode_locked": will_lock,
            "mode_detection_result": {
                "detected_mode": mode,
                "confidence": 0.7,
                "reason": "简单关键词检测后备逻辑",
                "fallback_used": True
            }
        }
    
    def chat_therapist_response(self, state: InterviewState) -> InterviewState:
        """CBT疗愈师响应 - 当用户想要闲聊时提供心理支持"""
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        latest_user_message = user_messages[-1].content if user_messages else ""
        
        # 构建CBT疗愈师的系统提示
        cbt_system_prompt = """你是灵溪智伴，一位专业的认知行为疗法(CBT)心理疗愈师。
        
你的任务是：
1. 生成温暖、理解和支持性的对话
2. 运用CBT技巧帮助用户认识和改变消极思维模式
3. 引导用户探索情感和行为模式
4. 提供实用的应对策略和技巧
5. 创造安全、非批判的对话环境

对话风格：
- 温暖、共情、专业
- 问开放性问题引导思考
- 适时提供CBT技巧和策略
- 鼓励用户表达内心感受
- 帮助用户建立积极的认知模式

请注意：
- 这是支持性闲聊，不是正式的问诊
- 如果用户提到严重的心理健康问题，建议寻求专业帮助
- 保持对话的轻松和支持性
- 不用生成开头的感情表情的标签，直接生成对话内容"""
        
        # 如果是评估完成后的闲聊
        if state.get("conversation_mode") == "assessment_complete":
            cbt_system_prompt += """
            
            特别提醒：用户刚刚完成了心理健康评估，现在进入闲聊环节。请：
            1. 感谢用户的参与和配合
            2. 确认评估已完成
            3. 提供后续的心理支持和建议
            4. 询问用户是否有其他想聊的话题
            """
        
        try:
            # 生成CBT疗愈师的回复
            cbt_prompt = f"用户说：{latest_user_message}"
            
            print("=" * 50)
            print("🔍 DEBUG - CBT_THERAPIST_RESPONSE LLM CALL")
            print("SYSTEM PROMPT:")
            print(cbt_system_prompt)
            print("USER PROMPT:")
            print(cbt_prompt)
            print("=" * 50)
            
            cbt_response = self.llm.invoke([
                SystemMessage(content=cbt_system_prompt),
                HumanMessage(content=cbt_prompt)
            ])
            
            print("RESPONSE:")
            print(cbt_response.content)
            print("=" * 50)
            
            final_response = cbt_response.content
            
            # 如果是评估完成状态，添加感谢和确认信息
            if state.get("conversation_mode") == "assessment_complete" and "感谢" not in final_response:
                completion_message = """感谢您的参与和配合！我已经完成了对您的心理健康评估。现在我们可以进行轻松的交流和心理支持。如果您有任何想聊的话题或需要心理支持，我很乐意陪伴您。
                """
                final_response = completion_message + "\n\n" + final_response
            
            return {
                **state,
                "final_response": final_response,
                "chat_therapist_active": True
            }
            
        except Exception as e:
            print(f"CBT疗愈师响应生成失败: {e}")
            # 后备响应
            fallback_response = """
            作为您的心理支持伙伴，我很高兴能和您聊天。您想聊什么呢？
            
            我可以：
            - 倾听您的感受和想法
            - 提供情绪支持和理解
            - 分享一些心理健康的小技巧
            - 陪您探讨生活中的各种话题
            
            请告诉我，您今天感觉如何？
            """
            
            return {
                **state,
                "final_response": fallback_response,
                "chat_therapist_active": True
            }