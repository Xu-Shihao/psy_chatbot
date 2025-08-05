"""
对话模式处理模块
包含对话模式检测和CBT疗愈师响应功能
"""

import json
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from agent import InterviewState
from workflow import EnhancedIntentDetector


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
            print(f"🔒 结构化模式强制锁定问诊流程 - 评估未完成")
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
            # 默认切换到CBT闲聊模式（没有明确问诊意图时）
            mode = "chat"
            will_lock = False
            print(f"🔄 简单检测：没有明确意图，切换到CBT闲聊模式", flush=True)
        
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
        
        # 更新对话历史 - 添加上一轮AI回复和当前用户消息（类似 fallback_response）
        updated_history = state.get("conversation_history", []).copy()
        
        # 添加上一轮的AI回复（如果存在且不是初始介绍）
        messages = state.get("messages", [])
        
        print(f"🔍 DEBUG - CBT Messages数量: {len(messages)}", flush=True)
        for i, msg in enumerate(messages):
            print(f"🔍 DEBUG - CBT Message[{i}]: {type(msg).__name__}", flush=True)
        
        # 只有当有真正的对话历史时，才添加AI回复
        if len(messages) > 3:  # SystemMessage + AIMessage(intro) + HumanMessage + AIMessage(real_response)
            last_ai_message = None
            # 从后往前找，跳过可能的初始介绍
            for msg in reversed(messages[2:]):  # 跳过前两个消息（SystemMessage + 初始介绍）
                if isinstance(msg, AIMessage):
                    last_ai_message = msg.content.strip()
                    break
            if last_ai_message:
                updated_history.append(f"You: {last_ai_message}")
                print(f"🔍 DEBUG - CBT添加AI历史: {last_ai_message[:50]}...", flush=True)
        
        # 添加当前用户消息
        updated_history.append(f"User: {latest_user_message}")
        print(f"🔍 DEBUG - CBT添加用户消息: {latest_user_message}", flush=True)
        
        # 获取最近20轮的历史记录用于prompt
        recent_history = updated_history[-20:] if len(updated_history) > 0 else []
        history_context = "\n".join(recent_history) if recent_history else "无对话历史"
        
        # 使用统一的prompt模板
        from prompts import PromptTemplates
        cbt_system_prompt = PromptTemplates.CBT_THERAPIST_SYSTEM_PROMPT
        
        # 如果是评估完成后的闲聊
        if state.get("conversation_mode") == "assessment_complete":
            cbt_system_prompt += PromptTemplates.CBT_ASSESSMENT_COMPLETE_ADDON
        
        try:
            # 生成包含历史对话的CBT疗愈师回复
            cbt_prompt = f"""## 最近对话历史：
{history_context}

请基于对话历史，生成合适的CBT疗愈师回应。"""
            
            print("=" * 50)
            print("🔍 DEBUG - CBT_THERAPIST_RESPONSE LLM CALL", flush=True)
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
            
            # 创建 AI 回复消息并添加到 messages 中（类似 fallback_response）
            ai_response_message = AIMessage(content=final_response)
            
            # 更新状态，包含更新后的对话历史和消息
            updated_state = state.copy()
            updated_state["conversation_history"] = updated_history
            updated_state["messages"] = state.get("messages", []) + [ai_response_message]
            updated_state["final_response"] = final_response
            updated_state["chat_therapist_active"] = True
            
            print(f"🔍 DEBUG - CBT添加AI回复到messages，新的messages数量: {len(updated_state['messages'])}", flush=True)
            print(f"🔍 DEBUG - CBT历史记录条数: {len(updated_history)}", flush=True)
            
            return updated_state
            
        except Exception as e:
            print(f"CBT疗愈师响应生成失败: {e}")
            # 后备响应
            fallback_response = PromptTemplates.CBT_FALLBACK_RESPONSE
            
            return {
                **state,
                "final_response": fallback_response,
                "chat_therapist_active": True
            }