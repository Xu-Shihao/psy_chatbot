"""
对话模式处理模块
包含对话模式检测和CBT疗愈师响应功能
"""

import json
from langchain.schema import HumanMessage, SystemMessage

from agent_types import InterviewState


class ConversationModeHandler:
    """对话模式处理器"""
    
    def __init__(self, llm, workflow_mode: str = "adaptive"):
        self.llm = llm
        self.workflow_mode = workflow_mode
    
    def detect_conversation_mode(self, state: InterviewState) -> InterviewState:
        """检测对话模式 - 在前5轮检测用户意图，一旦进入问诊模式就锁定"""
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
                "interview_mode_locked": True,  # 结构化模式直接锁定问诊
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
        
        # 如果超过5轮且未进入问诊模式，自动锁定为问诊模式
        if current_turn >= 5 and state.get("conversation_mode", "idle") != "interview":
            return {
                **state,
                "conversation_mode": "interview",
                "chat_therapist_active": False,
                "conversation_turn_count": current_turn + 1,
                "interview_mode_locked": True,
                "mode_detection_result": {
                    "detected_mode": "interview",
                    "confidence": 1.0,
                    "reason": "已达5轮对话，自动锁定为问诊模式"
                }
            }
        
        # 前5轮进行智能意图检测
        if current_turn < 5:
            # 使用LLM进行模式检测
            detection_prompt = f"""
            请分析用户的输入，判断用户是想要进行心理健康问诊还是想要闲聊：

            用户输入："{latest_user_message}"
            当前是第{current_turn + 1}轮对话

            判断标准：
            1. 如果用户明确表达想要闲聊、聊天、谈心等，则为"chat"模式
            2. 如果用户提到了具体的精神健康症状、心理问题、情绪困扰等，则为"interview"模式
            3. 如果用户正在回答问诊问题，则为"continue_interview"模式
            4. 如果用户说"开始评估"或类似的表达，则为"interview"模式

            请以JSON格式回复：
            {{
                "mode": "chat/interview/continue_interview",
                "confidence": 0.0-1.0,
                "reason": "判断理由",
                "key_indicators": ["关键指标1", "关键指标2"]
            }}
            """
            
            try:
                print("=" * 50)
                print("🔍 DEBUG - DETECT_CONVERSATION_MODE LLM CALL")
                print("PROMPT:")
                print(detection_prompt)
                print("=" * 50)
                
                detection_response = self.llm.invoke([
                    SystemMessage(content="你是一个专业的心理健康对话分析师，能够准确判断用户的对话意图。"),
                    HumanMessage(content=detection_prompt)
                ])
                
                print("RESPONSE:")
                print(detection_response.content)
                print("=" * 50)
                
                detection_result = json.loads(detection_response.content)
                detected_mode = detection_result["mode"]
                
                # 如果检测到问诊模式，立即锁定
                interview_will_lock = detected_mode in ["interview", "continue_interview"]
                
                # 根据当前状态调整检测结果
                current_mode = state.get("conversation_mode", "idle")
                if current_mode == "interview" and detected_mode == "interview":
                    detected_mode = "continue_interview"
                
                return {
                    **state,
                    "conversation_mode": detected_mode,
                    "chat_therapist_active": detected_mode == "chat",
                    "conversation_turn_count": current_turn + 1,
                    "interview_mode_locked": interview_will_lock,
                    "mode_detection_result": {
                        **detection_result,
                        "turn_count": current_turn + 1,
                        "locked": interview_will_lock
                    }
                }
                
            except Exception as e:
                print(f"模式检测失败: {e}")
                # 后备逻辑：简单关键词检测
                chat_keywords = ["闲聊", "聊天", "谈心", "聊聊", "随便聊", "陪我聊"]
                interview_keywords = ["抑郁", "焦虑", "失眠", "情绪", "心理", "症状", "困扰", "问题"]
                
                if any(keyword in latest_user_message for keyword in chat_keywords):
                    mode = "chat"
                    will_lock = False
                elif any(keyword in latest_user_message for keyword in interview_keywords):
                    mode = "interview"
                    will_lock = True
                else:
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
                        "confidence": 0.6,
                        "reason": "使用关键词检测后备逻辑",
                        "key_indicators": [],
                        "turn_count": current_turn + 1,
                        "locked": will_lock
                    }
                }
        
        # 超过5轮的情况（理论上不应该到达这里，因为上面已经处理了）
        return {
            **state,
            "conversation_mode": "interview",
            "chat_therapist_active": False,
            "conversation_turn_count": current_turn + 1,
            "interview_mode_locked": True,
            "mode_detection_result": {
                "detected_mode": "interview",
                "confidence": 1.0,
                "reason": "超过5轮对话，强制锁定为问诊模式"
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