"""
工作流管理模块
整合了所有工作流相关的功能，包括：
- LangGraph工作流构建
- 对话模式检测和处理
- 问诊流程管理
- CBT疗愈师响应
- 紧急情况处理
- 响应生成
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

from agent import InterviewState
from prompts import PromptTemplates, KeywordLibrary
from scid5_knowledge import scid5_kb


# =========================
# 工作流构建器
# =========================

class WorkflowBuilder:
    """工作流构建器"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def create_workflow(self) -> StateGraph:
        """创建问诊工作流"""
        workflow = StateGraph(InterviewState)
        
        # 添加节点
        workflow.add_node("start_interview", self.agent.start_interview)
        workflow.add_node("detect_conversation_mode", self.agent.detect_conversation_mode)
        workflow.add_node("chat_therapist_response", self.agent.chat_therapist_response)
        workflow.add_node("understand_and_respond", self.agent.understand_and_respond)
        workflow.add_node("ask_question", self.agent.ask_question)
        workflow.add_node("check_emergency", self.agent.check_emergency)
        workflow.add_node("generate_summary", self.agent.generate_summary)
        workflow.add_node("emergency_response", self.agent.emergency_response)
        
        # 设置入口点
        workflow.set_entry_point("start_interview")
        
        # 定义边
        workflow.add_edge("start_interview", "detect_conversation_mode")
        workflow.add_conditional_edges(
            "detect_conversation_mode",
            self.should_continue_after_mode_detection,
            {
                "interview": "understand_and_respond",  # 直接进入理解和回应阶段
                "chat": "chat_therapist_response",
                "assessment_complete": "chat_therapist_response"
            }
        )
        workflow.add_edge("chat_therapist_response", END)
        workflow.add_conditional_edges(
            "understand_and_respond",
            self.should_continue_after_understand_and_respond,
            {
                "continue": "check_emergency",
                "emergency": "emergency_response",
                "complete": "generate_summary",
                "wait_response": END
            }
        )
        workflow.add_conditional_edges(
            "check_emergency",
            self.should_continue_after_check,
            {
                "continue": "detect_conversation_mode",
                "complete": "generate_summary",
                "emergency": "emergency_response"
            }
        )
        workflow.add_edge("generate_summary", END)
        workflow.add_edge("emergency_response", END)
        
        return workflow
    
    def should_continue_after_mode_detection(self, state: InterviewState) -> str:
        """决定模式检测后的下一步"""
        # 如果是结构化模式且未完成评估，强制进入问诊流程
        if (self.agent.workflow_mode == "structured" and 
            not state.get("assessment_complete", False)):
            print(f"🔒 工作流强制路由到问诊模式 - 结构化模式未完成评估")
            return "interview"
        
        # 智能检测模式的原有逻辑
        mode = state.get("conversation_mode", "interview")
        
        if mode == "chat":
            return "chat"
        elif mode in ["interview", "continue_interview"]:
            return "interview"  # 统一映射到interview流程
        elif mode == "assessment_complete":
            return "assessment_complete"  
        else:
            return "interview"  # 默认进入问诊模式
    
    def should_continue_after_understand_and_respond(self, state: InterviewState) -> str:
        """理解和回应后的流程控制"""
        if state["emergency_situation"]:
            return "emergency"
        elif state["assessment_complete"]:
            return "complete"
        else:
            return "continue"
    
    def should_continue_after_question(self, state: InterviewState) -> str:
        """问题后的流程控制（保留用于兼容性）"""
        if state["emergency_situation"]:
            return "emergency"
        return "wait_response"
    
    def should_continue_after_check(self, state: InterviewState) -> str:
        """检查后的流程控制"""
        if state["emergency_situation"]:
            return "emergency"
        elif state["assessment_complete"]:
            return "complete"
        else:
            # 获取下一个问题
            current_question_id = state["current_question_id"]
            last_response = ""
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'content'):
                    last_response = msg.content
                    break
            
            if current_question_id and last_response:
                next_question = scid5_kb.get_next_question(current_question_id, last_response)
                if next_question:
                    # 更新到下一个问题
                    state["current_question_id"] = next_question.id
                    return "continue"
                else:
                    state["assessment_complete"] = True
                    return "complete"
            
            return "complete"


# =========================
# 增强意图检测器
# =========================

class EnhancedIntentDetector:
    """增强的意图检测器"""
    
    def __init__(self, llm, workflow_mode: str = "adaptive"):
        self.llm = llm
        self.workflow_mode = workflow_mode
    
    def detect_conversation_mode(self, state: InterviewState) -> InterviewState:
        """增强的对话模式检测"""
        
        # 1. 检查基本状态
        if state.get("assessment_complete", False):
            return self._handle_assessment_complete(state)
        
        # 2. 检查紧急情况
        emergency_result = self._check_emergency_priority(state)
        if emergency_result["is_emergency"]:
            return self._handle_emergency_mode(state, emergency_result)
        
        # 3. 获取上下文信息
        context = self._extract_conversation_context(state)
        
        # 4. 结构化模式处理
        if self.workflow_mode == "structured":
            return self._handle_structured_mode(state, context)
        
        # 5. 智能检测模式
        return self._handle_adaptive_mode(state, context)
    
    def _extract_conversation_context(self, state: InterviewState) -> Dict:
        """提取对话上下文信息"""
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
        
        if not user_messages:
            return {
                "current_turn": 0,
                "latest_message": "",
                "message_length": 0,
                "conversation_history": [],
                "emotional_state": "neutral",
                "symptom_severity": "none",
                "user_engagement": "unknown"
            }
        
        latest_message = user_messages[-1].content
        current_turn = state.get("conversation_turn_count", 0)
        
        # 分析情绪状态
        emotional_state = self._detect_emotional_state(latest_message)
        
        # 分析症状严重程度
        symptom_severity = self._assess_symptom_severity(latest_message)
        
        # 评估用户参与度
        user_engagement = self._assess_user_engagement(state, latest_message)
        
        # 分析消息复杂度
        message_complexity = self._analyze_message_complexity(latest_message)
        
        return {
            "current_turn": current_turn,
            "latest_message": latest_message,
            "message_length": len(latest_message),
            "conversation_history": [msg.content for msg in user_messages[-3:]],  # 最近3条
            "emotional_state": emotional_state,
            "symptom_severity": symptom_severity,
            "user_engagement": user_engagement,
            "message_complexity": message_complexity,
            "response_pattern": self._analyze_response_pattern(state)
        }
    
    def _detect_emotional_state(self, message: str) -> str:
        """检测用户的情绪状态"""
        for emotion, keywords in KeywordLibrary.EMOTION_PATTERNS.items():
            if any(keyword in message for keyword in keywords):
                return emotion
        return "neutral"
    
    def _assess_symptom_severity(self, message: str) -> str:
        """评估症状严重程度"""
        for severity, keywords in KeywordLibrary.SYMPTOM_KEYWORDS.items():
            if any(keyword in message for keyword in keywords):
                return severity.replace("_severity", "")
        return "none"
    
    def _assess_user_engagement(self, state: InterviewState, latest_message: str) -> str:
        """评估用户参与度"""
        # 基于消息长度、回复速度、内容详细程度评估
        message_length = len(latest_message)
        
        if message_length < 5:
            return "low"
        elif message_length < 20:
            return "medium"
        else:
            return "high"
    
    def _analyze_message_complexity(self, message: str) -> Dict:
        """分析消息复杂度"""
        return {
            "length": len(message),
            "word_count": len(message.split()),
            "sentence_count": len([s for s in message.split('。') if s.strip()]),
            "question_count": message.count('？') + message.count('?'),
            "has_specific_details": len(message) > 30 and ('因为' in message or '所以' in message)
        }
    
    def _analyze_response_pattern(self, state: InterviewState) -> str:
        """分析用户的回复模式"""
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if len(user_messages) < 2:
            return "initial"
        
        recent_messages = [msg.content for msg in user_messages[-3:]]
        avg_length = sum(len(msg) for msg in recent_messages) / len(recent_messages)
        
        if avg_length < 10:
            return "brief"
        elif avg_length > 50:
            return "detailed"
        else:
            return "moderate"
    
    def _check_emergency_priority(self, state: InterviewState) -> Dict:
        """检查是否有紧急情况需要优先处理"""
        user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        if not user_messages:
            return {"is_emergency": False, "severity": "none"}
        
        latest_message = user_messages[-1].content
        
        # 检查高风险关键词
        for keyword in KeywordLibrary.SYMPTOM_KEYWORDS["high_severity"]:
            if keyword in latest_message:
                return {
                    "is_emergency": True,
                    "severity": "high",
                    "trigger_keywords": [keyword],
                    "message": latest_message
                }
        
        return {"is_emergency": False, "severity": "none"}
    
    def _handle_emergency_mode(self, state: InterviewState, emergency_result: Dict) -> InterviewState:
        """处理紧急情况模式"""
        return {
            **state,
            "conversation_mode": "emergency",
            "emergency_situation": True,
            "chat_therapist_active": False,
            "interview_mode_locked": True,
            "mode_detection_result": {
                "detected_mode": "emergency",
                "confidence": 1.0,
                "reason": f"检测到紧急情况：{emergency_result['trigger_keywords']}",
                "emergency_details": emergency_result
            }
        }
    
    def _handle_assessment_complete(self, state: InterviewState) -> InterviewState:
        """处理评估完成状态"""
        return {
            **state,
            "conversation_mode": "assessment_complete",
            "chat_therapist_active": True,
            "mode_detection_result": {
                "detected_mode": "assessment_complete",
                "confidence": 1.0,
                "reason": "评估已完成，进入后续支持性对话模式"
            }
        }
    
    def _handle_structured_mode(self, state: InterviewState, context: Dict) -> InterviewState:
        """处理结构化模式"""
        current_turn = context["current_turn"]
        
        # 结构化模式：强制问诊直到完成
        return {
            **state,
            "conversation_mode": "continue_interview",
            "chat_therapist_active": False,
            "conversation_turn_count": current_turn + 1,
            "interview_mode_locked": True,
            "mode_detection_result": {
                "detected_mode": "continue_interview",
                "confidence": 1.0,
                "reason": "结构化模式：强制进行完整问诊流程",
                "context_analysis": context
            }
        }
    
    def _handle_adaptive_mode(self, state: InterviewState, context: Dict) -> InterviewState:
        """处理自适应模式"""
        current_turn = context["current_turn"]
        latest_message = context["latest_message"]
        emotional_state = context["emotional_state"]
        symptom_severity = context["symptom_severity"]
        user_engagement = context["user_engagement"]
        
        # 检查是否已锁定问诊模式
        if state.get("interview_mode_locked", False):
            return {
                **state,
                "conversation_mode": "continue_interview",
                "chat_therapist_active": False,
                "conversation_turn_count": current_turn + 1,
                "mode_detection_result": {
                    "detected_mode": "continue_interview",
                    "confidence": 1.0,
                    "reason": "问诊模式已锁定，继续完成评估",
                    "context_analysis": context
                }
            }
        
        # 使用增强的LLM检测
        detection_result = self._enhanced_llm_detection(state, context)
        
        # 基于多维度分析做最终决策
        final_mode = self._make_final_decision(detection_result, context, state)
        
        return final_mode
    
    def _enhanced_llm_detection(self, state: InterviewState, context: Dict) -> Dict:
        """增强的LLM意图检测"""
        latest_message = context["latest_message"]
        current_turn = context["current_turn"]
        emotional_state = context["emotional_state"]
        symptom_severity = context["symptom_severity"]
        conversation_history = context["conversation_history"]
        
        # 构建更智能的检测提示
        detection_prompt = PromptTemplates.get_enhanced_intent_detection_prompt(
            latest_message, current_turn, emotional_state, symptom_severity, conversation_history
        )
        
        try:
            print("🧠 增强意图检测 - LLM分析中...")
            
            detection_response = self.llm.invoke([
                SystemMessage(content=PromptTemplates.INTENT_ANALYST_SYSTEM_PROMPT),
                HumanMessage(content=detection_prompt)
            ])
            
            result = json.loads(detection_response.content)
            print(f"✅ 检测结果：{result['primary_intent']} (置信度: {result['confidence']})")
            
            return result
            
        except Exception as e:
            print(f"❌ 增强检测失败，使用后备逻辑: {e}")
            return self._fallback_detection(latest_message, context)
    
    def _fallback_detection(self, message: str, context: Dict) -> Dict:
        """后备检测逻辑"""
        symptom_severity = context["symptom_severity"]
        emotional_state = context["emotional_state"]
        
        # 基于症状严重程度和情绪状态的后备判断
        if symptom_severity in ["high", "medium"]:
            return {
                "primary_intent": "interview",
                "confidence": 0.8,
                "reasoning": f"检测到{symptom_severity}级别症状，建议问诊评估",
                "urgency_level": "high" if symptom_severity == "high" else "medium"
            }
        elif emotional_state in ["distressed", "anxious", "sad"]:
            return {
                "primary_intent": "supportive_chat",
                "confidence": 0.7,
                "reasoning": f"用户情绪状态为{emotional_state}，需要支持性对话",
                "urgency_level": "medium"
            }
        elif any(keyword in message for keyword in KeywordLibrary.CHAT_KEYWORDS):
            return {
                "primary_intent": "chat",
                "confidence": 0.8,
                "reasoning": "用户明确表达想要闲聊",
                "urgency_level": "low"
            }
        else:
            return {
                "primary_intent": "continue_interview",
                "confidence": 0.6,
                "reasoning": "默认继续问诊流程",
                "urgency_level": "low"
            }
    
    def _make_final_decision(self, detection_result: Dict, context: Dict, state: InterviewState) -> InterviewState:
        """基于多维度分析做最终决策"""
        primary_intent = detection_result["primary_intent"]
        confidence = detection_result.get("confidence", 0.5)
        urgency_level = detection_result.get("urgency_level", "low")
        current_turn = context["current_turn"]
        
        # 决策逻辑优化
        final_mode = primary_intent
        should_lock = False
        chat_active = False
        
        # 高置信度且高紧急度的问诊需求，立即锁定
        if primary_intent in ["interview", "continue_interview"]:
            if confidence > 0.7 or urgency_level == "high":
                should_lock = True
            chat_active = False
            
        elif primary_intent in ["chat", "supportive_chat"]:
            chat_active = True
            should_lock = False
            
        # 对于低置信度的判断，默认切换到CBT闲聊模式
        if confidence < 0.6 and current_turn < 3:
            final_mode = "chat"  # 没有明确问诊意图时，默认进入CBT闲聊模式
            chat_active = True
            should_lock = False
            print(f"🔄 意图不明确（置信度: {confidence:.2f}），切换到CBT闲聊模式", flush=True)
            
        return {
            **state,
            "conversation_mode": final_mode,
            "chat_therapist_active": chat_active,
            "conversation_turn_count": current_turn + 1,
            "interview_mode_locked": should_lock,
            "mode_detection_result": {
                **detection_result,
                "final_decision": final_mode,
                "locked": should_lock,
                "turn_count": current_turn + 1,
                "context_analysis": context
            }
        }
    
    def get_detection_summary(self, state: InterviewState) -> str:
        """获取检测结果摘要"""
        result = state.get("mode_detection_result", {})
        
        summary = f"""
        🔍 意图检测结果：
        - 检测模式：{result.get('detected_mode', '未知')}
        - 置信度：{result.get('confidence', 0):.2f}
        - 原因：{result.get('reason', '无')}
        - 情感需求：{result.get('emotional_needs', '未分析')}
        - 紧急程度：{result.get('urgency_level', '未知')}
        """
        
        return summary


# =========================
# 对话模式处理器
# =========================

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
        if any(keyword in latest_user_message for keyword in KeywordLibrary.CHAT_KEYWORDS):
            mode = "chat"
            will_lock = False
        elif any(keyword in latest_user_message for keyword in KeywordLibrary.INTERVIEW_KEYWORDS):
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



# =========================
# 问诊流程处理器
# =========================

class InterviewFlowHandler:
    """问诊流程处理器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def start_interview(self, state: InterviewState) -> InterviewState:
        """开始问诊 - 生成自我介绍和引导开场白"""
        system_message = SystemMessage(content=PromptTemplates.COUNSELOR_SYSTEM_PROMPT)
        
        # 生成自我介绍和引导开场白
        try:
            print("=" * 50)
            print("🔍 DEBUG - START_INTERVIEW LLM CALL", flush=True)
            print("PROMPT:")
            print(PromptTemplates.INITIAL_INTERVIEW_PROMPT)
            print("=" * 50)
            
            response = self.llm.invoke([HumanMessage(content=PromptTemplates.INITIAL_INTERVIEW_PROMPT)])
            initial_response = response.content.strip()
            
            print("RESPONSE:")
            print(initial_response)
            print("=" * 50)
            
            # 添加重要说明
            initial_response += PromptTemplates.IMPORTANT_DISCLAIMER
            
        except Exception as e:
            print(f"🔍 DEBUG: LLM生成开头失败: {e}", flush=True)
            # 如果LLM调用失败，使用备用开头
            initial_response = PromptTemplates.FALLBACK_INTERVIEW_INTRO + PromptTemplates.IMPORTANT_DISCLAIMER
        
        # 创建AI消息
        intro_message = AIMessage(content=initial_response)
        
        return {
            **state,
            "messages": [system_message, intro_message],
            "current_question_id": "initial",  # 设置为初始状态，等待用户分享主诉
            "user_responses": {},
            "assessment_complete": False,
            "emergency_situation": False,
            "summary": "",
            "needs_followup": False,
            "conversation_history": [],
            "followup_questions": [],
            "risk_level": "low",
            "risk_indicators": [],
            "emotional_state": "",
            "content_complete": False,
            "understanding_summary": "",
            "empathetic_response": "",
            "current_analysis": {},
            "debug_info": [],
            "session_start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question_sequence": ["initial"],
            "final_response": "",
            "assessment_duration": 0,
            "user_engagement_level": "medium",
            "response_detail_level": "moderate",
            "symptoms_identified": [],
            "domains_assessed": ["initial"],
            "severity_indicators": {},
            "chief_complaint": "",
            "conversation_mode": "idle",  # 开始时为空闲模式，等待用户意图检测
            "chat_therapist_active": False,
            "mode_detection_result": {},
            "conversation_turn_count": 0,  # 初始化会话轮数
            "interview_mode_locked": False,  # 初始化问诊模式锁定状态
            "current_topic": "初始化问诊",
            "is_follow_up": False,
            "assessed_criteria": {},  # 初始化已评估的诊断标准（只记录已问过的症状）
            "current_disorder_focus": "mood_disorders"  # 初始关注抑郁症
        }
    
    def understand_and_respond(self, state: InterviewState) -> InterviewState:
        """理解用户回答并提供共情回应，同时提出下一个问题"""
        if len(state["messages"]) < 2:
            return state
        
        # 获取最后一条用户消息
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break
        
        if not last_user_message:
            return state
        
        current_question_id = state["current_question_id"]
        current_question = scid5_kb.questions.get(current_question_id) if current_question_id else None
        
        # 获取对话上下文
        conversation_context = "\n".join(state["conversation_history"][-10:]) if state["conversation_history"] else ""
        
        # 构建综合分析和回应的提示
        next_question_info = self._get_next_question_info(current_question_id, last_user_message, state)
        
        current_disorder_focus = state.get("current_question_id", "depression_screening")
        
        # 获取上一轮AI的真实回复作为上下文
        last_ai_response = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                last_ai_response = msg.content
                break
        
        comprehensive_prompt = PromptTemplates.get_comprehensive_analysis_prompt(
            conversation_context, last_ai_response, last_user_message, 
            current_disorder_focus, next_question_info
        )
        
        print("=" * 50)
        print("🔍 DEBUG - UNDERSTAND_AND_RESPOND (COMPREHENSIVE) LLM CALL", flush=True)
        print("PROMPT:")
        print(comprehensive_prompt)
        print("=" * 50)
        
        response = self.llm.invoke([HumanMessage(content=comprehensive_prompt)])
        
        print("RESPONSE:")
        print(response.content)
        print("=" * 50)
        
        try:
            analysis = json.loads(response.content)
        except:
            # 如果JSON解析失败，提供默认回应
            analysis = {
                "emotional_state": "未能识别",
                "risk_level": "low",
                "risk_indicators": [],
                "understanding_summary": last_user_message,
                "has_next_question": False,
                "next_question_id": None,
                "comprehensive_response": "谢谢您的分享。我听到了您的感受，让我们继续聊聊。",
                "assessment_complete": False
            }
        
        # 记录用户回答
        updated_responses = state["user_responses"].copy()
        if current_question_id:
            updated_responses[current_question_id] = last_user_message
        
        # 更新已评估的症状标准
        updated_state_with_criteria = self._update_assessed_criteria(state, current_question_id, last_user_message)
        
        # 生成回应消息
        ai_response = AIMessage(content=analysis["comprehensive_response"])
        
        # 更新对话历史 - 使用真实的AI回复内容
        updated_history = state["conversation_history"].copy()
        # 如果是第一轮对话，添加上一轮的AI回复（如果存在）
        if len(state["messages"]) > 1:
            last_ai_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg.content
                    break
            if last_ai_message:
                updated_history.append(f"You: {last_ai_message}")
        updated_history.append(f"User: {last_user_message}")
        # 注意：不在这里添加当前AI回复，因为它会在下一轮的历史中显示
        
        # 更新当前问题ID
        next_question_id = analysis.get("next_question_id")
        if next_question_id == "assessment_complete":
            next_question_id = None
            analysis["assessment_complete"] = True
        
        # 根据next_question_id动态更新current_disorder_focus
        updated_disorder_focus = state.get("current_disorder_focus", "mood_disorders")
        if next_question_id:
            if next_question_id.startswith("anxiety"):
                updated_disorder_focus = "anxiety_disorders"
            elif next_question_id.startswith("depression"):
                updated_disorder_focus = "mood_disorders"
            elif next_question_id.startswith("ocd"):
                updated_disorder_focus = "obsessive_compulsive"
            elif next_question_id.startswith("ptsd"):
                updated_disorder_focus = "trauma_related"
            elif next_question_id.startswith("psychotic"):
                updated_disorder_focus = "psychotic_disorders"
        
        # 根据下一个问题ID更新当前话题
        topic_mapping = {
            "depression_screening": "抑郁症筛查",
            "anxiety_screening": "焦虑症筛查",
            "ocd_screening": "强迫症筛查",
            "ptsd_screening": "创伤后应激障碍筛查",
            "psychotic_screening": "精神病性障碍筛查",
            "initial": "初始问诊",
            "initial_screening": "初始筛查"
        }
        
        updated_topic = topic_mapping.get(next_question_id, state.get("current_topic", "问诊中"))
        if analysis.get("assessment_complete", False):
            updated_topic = "评估已完成"
        
        return {
            **updated_state_with_criteria,
            "messages": state["messages"] + [ai_response],
            "user_responses": updated_responses,
            "conversation_history": updated_history,
            "emergency_situation": analysis["risk_level"] == "high",
            "risk_level": analysis["risk_level"],
            "risk_indicators": analysis.get("risk_indicators", []),
            "emotional_state": analysis["emotional_state"],
            "understanding_summary": analysis["understanding_summary"],
            "current_analysis": analysis,
            "current_question_id": next_question_id,
            "current_disorder_focus": updated_disorder_focus,  # 应用更新的障碍关注类型
            "assessment_complete": analysis.get("assessment_complete", False),
            "needs_followup": False,  # 不再需要单独的追问流程
            "session_start_time": state["session_start_time"],
            "question_sequence": state["question_sequence"] + [current_question_id] if current_question_id else state["question_sequence"],
            "assessment_duration": state["assessment_duration"] + 1 if current_question_id else state["assessment_duration"],
            "user_engagement_level": "high" if analysis["risk_level"] == "high" else "medium",
            "response_detail_level": "detailed" if len(analysis.get("risk_indicators", [])) > 0 else "moderate",
            "symptoms_identified": state["symptoms_identified"] + ([current_question.text] if current_question else ["初始问诊"]),
            "domains_assessed": state["domains_assessed"] + ([current_question.category.value] if current_question else ["initial_screening"]),
            "severity_indicators": {**state["severity_indicators"], **({current_question.category.value: analysis["risk_level"]} if current_question else {"initial_screening": analysis["risk_level"]})},
            "chief_complaint": state["chief_complaint"] if state["chief_complaint"] else (last_user_message if current_question_id == "initial" else state["chief_complaint"]),
            "conversation_mode": "assessment_complete" if analysis.get("assessment_complete", False) else state["conversation_mode"],
            "chat_therapist_active": True if analysis.get("assessment_complete", False) else state["chat_therapist_active"],
            "mode_detection_result": state["mode_detection_result"],
            "current_topic": updated_topic,
            "is_follow_up": state["is_follow_up"]
        }
    
    def _get_next_question_info(self, current_question_id: str, user_response: str, state: InterviewState = None) -> str:
        """获取下一个问题的信息用于生成回应，基于诊断标准动态选择"""
        print("=" * 80)
        print("🔍 DEBUG - _get_next_question_info 方法开始", flush=True)
        print(f"📥 输入参数:")
        print(f"   current_question_id: {current_question_id}")
        print(f"   user_response: {user_response[:100]}..." if len(user_response) > 100 else f"   user_response: {user_response}")
        print(f"   state存在: {state is not None}")
        
        if not current_question_id or current_question_id == "initial":
            print("🔄 逻辑分支: 初始问诊阶段")
            result = "问题目的：开始心理健康评估，询问用户目前最主要的困扰，如果用户已经表达了困扰，则安慰并追问用户的主诉。"
            print(f"📤 返回结果: {result}")
            print("=" * 80)
            return result
        
        # 获取当前关注的障碍类型和对应的诊断标准
        current_disorder = state.get("current_disorder_focus", "mood_disorders") if state else "mood_disorders"
        assessed_criteria = state.get("assessed_criteria", {}) if state else {}
        
        print(f"📊 状态数据:")
        print(f"   current_disorder: {current_disorder}")
        print(f"   assessed_criteria: {assessed_criteria}")
        
        # 如果是初始问诊且用户提及了抑郁症状，优先安排抑郁症筛查
        if (not current_question_id or current_question_id == "initial") and user_response:
            print("🔄 逻辑分支: 检查初始问诊中的症状关键词")
            response_lower = user_response.lower()
            print(f"   用户回答(小写): {response_lower}")
            
            depression_keywords = ["抑郁", "郁闷", "低落", "沮丧", "悲伤", "难过"]
            anxiety_keywords = ["焦虑", "紧张", "担心", "恐慌", "害怕"]
            
            depression_found = [kw for kw in depression_keywords if kw in response_lower]
            anxiety_found = [kw for kw in anxiety_keywords if kw in response_lower]
            
            print(f"   抑郁关键词匹配: {depression_found}")
            print(f"   焦虑关键词匹配: {anxiety_found}")
            
            if depression_found:
                result = "下一步将进行抑郁症状评估，询问用户关于情绪低落的具体情况"
                print(f"📤 返回结果(抑郁症状): {result}")
                print("=" * 80)
                return result
            elif anxiety_found:
                result = "下一步将进行焦虑症状评估，询问用户关于焦虑担心的具体情况"
                print(f"📤 返回结果(焦虑症状): {result}")
                print("=" * 80)
                return result
        
        # 修复：将current_disorder映射到正确的类型
        disorder_type_mapping = {
            "anxiety_screening": "anxiety_disorders",
            "depression_screening": "mood_disorders", 
            "mood_disorders": "mood_disorders",
            "anxiety_disorders": "anxiety_disorders",
            "obsessive_compulsive": "obsessive_compulsive",
            "trauma_related": "trauma_related",
            "psychotic_disorders": "psychotic_disorders"
        }
        
        print(f"🔄 逻辑分支: 障碍类型映射")
        print(f"   disorder_type_mapping: {disorder_type_mapping}")
        
        # 获取对应障碍的诊断标准
        disorder_mapping = {
            "mood_disorders": "major_depression",
            "anxiety_disorders": "generalized_anxiety", 
            "psychotic_disorders": None  # 精神病性障碍暂时不使用详细标准
        }
        
        print(f"   disorder_mapping: {disorder_mapping}")
        
        # 修复：使用正确的disorder类型
        mapped_disorder_type = disorder_type_mapping.get(current_disorder, current_disorder)
        disorder_key = disorder_mapping.get(mapped_disorder_type)
        
        print(f"   mapped_disorder_type: {mapped_disorder_type}")
        print(f"   disorder_key: {disorder_key}")
        
        if disorder_key and disorder_key in scid5_kb.diagnostic_criteria:
            print(f"🔄 逻辑分支: 使用诊断标准 - {disorder_key}")
            criteria = scid5_kb.diagnostic_criteria[disorder_key]
            assessed_for_disorder = assessed_criteria.get(mapped_disorder_type, [])
            
            print(f"   诊断标准数量: {len(criteria.criteria)}")
            print(f"   所有标准: {criteria.criteria}")
            print(f"   已评估标准: {assessed_for_disorder}")
            
            # 找出尚未评估的症状标准
            remaining_criteria = [c for c in criteria.criteria if c not in assessed_for_disorder]
            print(f"   剩余待评估标准: {remaining_criteria}")
            
            # 根据尚未评估的症状标准生成具体问题
            if remaining_criteria:
                next_symptom = remaining_criteria[0]
                print(f"   选择下一个症状: {next_symptom}")
                
                # 根据症状标准生成对应的问题
                symptom_questions = {
                    "情绪低落": "在过去的两周内，您是否几乎每天都感到情绪低落、沮丧或绝望？",
                    "兴趣丧失": "在过去的两周内，您是否对平时感兴趣或愉快的活动明显失去兴趣？",
                    "食欲改变": "在过去的两周内，您的食欲是否有明显变化？比如食欲显著增加或减少，体重有明显变化吗？",
                    "睡眠障碍": "在过去的两周内，您的睡眠怎么样？是否有失眠、早醒或睡眠过多的情况？",
                    "精神运动改变": "在过去的两周内，您是否感到坐立不安，或者动作和思维比平时缓慢？",
                    "疲劳": "在过去的两周内，您是否几乎每天都感到疲劳或精力不足？",
                    "无价值感/罪恶感": "在过去的两周内，您是否经常感到自己没有价值，或者有过度的、不合理的罪恶感？",
                    "注意力问题": "在过去的两周内，您是否发现自己难以集中注意力，或者在做决定时犹豫不决？",
                    "死亡念头": "在过去的两周内，您是否反复想到死亡，或者有过伤害自己的想法？",
                    "过度焦虑和担心": "在过去的6个月内，您是否经常感到过度的担心或焦虑，难以控制这些担心？",
                    "坐立不安": "当您感到焦虑时，是否经常感到坐立不安或紧张不安？",
                    "容易疲劳": "您是否因为担心和焦虑而容易感到疲劳？",
                    "注意力集中困难": "焦虑是否影响了您集中注意力的能力？",
                    "易怒": "您是否发现自己比平时更容易生气或烦躁？",
                    "肌肉紧张": "您是否经常感到肌肉紧张或身体僵硬？"
                }
                
                question_text = symptom_questions.get(next_symptom, f"请告诉我关于{next_symptom}方面的情况")
                next_question_info = f"评估{next_symptom}"
                
                print(f"   对应问题文本: {question_text}")
                print(f"📤 返回结果: {next_question_info}")
                print("=" * 80)
                return next_question_info
            else:
                print("🔄 逻辑分支: 当前障碍评估完成")
                next_question_info = "评估已完成，生成summary报告并进入CBT疗愈模式"
                print(f"📤 返回结果: {next_question_info}")
                print("=" * 80)
                return next_question_info
        else:
            print(f"🔄 逻辑分支: 使用原有逻辑 (disorder_key: {disorder_key})")
            print(f"   scid5_kb.diagnostic_criteria 中包含的键: {list(scid5_kb.diagnostic_criteria.keys()) if hasattr(scid5_kb, 'diagnostic_criteria') else '无diagnostic_criteria'}")
            
            # 使用原有逻辑作为备选
            next_question = scid5_kb.get_next_question(current_question_id, user_response)
            print(f"   scid5_kb.get_next_question 返回: {next_question}")
            
            if next_question:
                next_question_info = f"{getattr(next_question, 'purpose', '评估相关症状')}"
                print(f"📤 返回结果(原有逻辑): {next_question_info}")
            else:
                next_question_info = "评估已完成，生成summary报告并进入CBT疗愈模式"
                print(f"📤 返回结果(评估完成): {next_question_info}")
            
            print("=" * 80)
            return next_question_info
    
    def _update_assessed_criteria(self, state: InterviewState, current_question_id: str, user_response: str) -> InterviewState:
        """更新已评估的症状标准 - 只记录已问过的问题，不做疾病判断"""
        if not current_question_id or current_question_id == "initial":
            return state
            
        # 根据问题ID和回答更新已评估的症状标准
        current_disorder = state.get("current_disorder_focus", "mood_disorders")
        assessed_criteria = state.get("assessed_criteria", {}).copy()
        
        # 初始化当前障碍的评估记录
        if current_disorder not in assessed_criteria:
            assessed_criteria[current_disorder] = []
            
        # 动态确定当前问题对应的症状标准
        current_symptom = self._get_current_symptom_from_question(current_disorder, assessed_criteria.get(current_disorder, []), current_question_id)
        
        if current_symptom:
            # 只记录已问过的症状标准，不做疾病判断
            if current_symptom not in assessed_criteria[current_disorder]:
                assessed_criteria[current_disorder].append(current_symptom)
                
                print(f"🔍 DEBUG - 更新已评估标准:", flush=True)
                print(f"   障碍类型: {current_disorder}")
                print(f"   新增症状标准: {current_symptom}")
                print(f"   当前已评估标准: {assessed_criteria[current_disorder]}")
        
        # 更新状态 - 移除criteria_results，疾病判断留给报告生成阶段
        updated_state = state.copy()
        updated_state["assessed_criteria"] = assessed_criteria
        
        return updated_state
    
    def _get_current_symptom_from_question(self, disorder_type: str, assessed_symptoms: list, question_id: str) -> str:
        """根据障碍类型和已评估症状，确定当前问题对应的症状标准"""
        # 障碍类型映射
        disorder_type_mapping = {
            "anxiety_screening": "anxiety_disorders",
            "depression_screening": "mood_disorders", 
            "mood_disorders": "mood_disorders",
            "anxiety_disorders": "anxiety_disorders",
            "obsessive_compulsive": "obsessive_compulsive",
            "trauma_related": "trauma_related",
            "psychotic_disorders": "psychotic_disorders"
        }
        
        # 获取对应障碍的诊断标准
        disorder_mapping = {
            "mood_disorders": "major_depression",
            "anxiety_disorders": "generalized_anxiety", 
            "psychotic_disorders": None
        }
        
        mapped_disorder_type = disorder_type_mapping.get(disorder_type, disorder_type)
        disorder_key = disorder_mapping.get(mapped_disorder_type)
        
        if disorder_key and disorder_key in scid5_kb.diagnostic_criteria:
            criteria = scid5_kb.diagnostic_criteria[disorder_key]
            
            # 找出尚未评估的症状标准
            remaining_criteria = [c for c in criteria.criteria if c not in assessed_symptoms]
            
            if remaining_criteria:
                # 返回下一个待评估的症状标准
                next_symptom = remaining_criteria[0]
                print(f"🔍 DEBUG - _get_current_symptom_from_question:", flush=True)
                print(f"   障碍类型: {disorder_type} -> {mapped_disorder_type}")
                print(f"   诊断标准: {disorder_key}")
                print(f"   所有标准: {criteria.criteria}")
                print(f"   已评估: {assessed_symptoms}")
                print(f"   剩余待评估: {remaining_criteria}")
                print(f"   当前症状: {next_symptom}")
                return next_symptom
        
        # 如果无法从诊断标准中确定，使用备用逻辑
        fallback_mapping = {
            "depression_screening": "情绪低落",
            "depression_interest": "兴趣丧失", 
            "anxiety_screening": "过度焦虑和担心",
            "panic_screening": "惊恐发作"
        }
        
        return fallback_mapping.get(question_id, "其他症状")
    
    def ask_question(self, state: InterviewState) -> InterviewState:
        """独立提问方法（保留用于特殊情况）"""
        # 这个方法现在主要用于特殊情况，大部分情况下understand_and_respond已经包含了提问
        return state


# =========================
# 紧急情况处理器
# =========================

class EmergencyHandler:
    """紧急情况处理器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def check_emergency(self, state: InterviewState) -> InterviewState:
        """检查紧急情况"""
        # 获取最近的用户回答
        last_response = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_response = msg.content.lower()
                break
        
        # 使用LLM进行更智能的风险评估
        risk_prompt = PromptTemplates.get_emergency_risk_assessment_prompt(
            last_response, state.get("current_question_id", "")
        )
        
        print("=" * 50)
        print("🔍 DEBUG - CHECK_EMERGENCY LLM CALL", flush=True)
        print("PROMPT:")
        print(risk_prompt)
        print("=" * 50)
        
        response = self.llm.invoke([HumanMessage(content=risk_prompt)])
        
        print("RESPONSE:")
        print(response.content)
        print("=" * 50)
        
        try:
            risk_analysis = json.loads(response.content)
            emergency_detected = risk_analysis.get("immediate_action_needed", False)
        except:
            # 如果LLM分析失败，使用关键词检测作为备选
            emergency_detected = any(keyword in last_response for keyword in KeywordLibrary.EMERGENCY_KEYWORDS)
        
        return {
            **state,
            "emergency_situation": emergency_detected
        }
    
    def emergency_response(self, state: InterviewState) -> InterviewState:
        """处理紧急情况"""
        emergency_message = AIMessage(content=PromptTemplates.EMERGENCY_RESPONSE_MESSAGE)
        
        return {
            **state,
            "messages": state["messages"] + [emergency_message],
            "assessment_complete": True
        }


# =========================
# 响应生成器
# =========================

class ResponseGenerator:
    """回应生成器"""
    
    def __init__(self, llm):
        self.llm = llm
        self.workflow_mode = "adaptive"  # 默认模式
    
    def generate_summary(self, state: InterviewState) -> InterviewState:
        """生成评估总结"""
        assessment_summary = scid5_kb.generate_assessment_summary()
        
        # 使用LLM生成更详细和个性化的总结
        prompt = PromptTemplates.get_assessment_summary_prompt(
            state["conversation_history"], 
            state["user_responses"], 
            assessment_summary
        )
        
        print("=" * 50)
        print("🔍 DEBUG - GENERATE_SUMMARY LLM CALL", flush=True)
        print("PROMPT:")
        print(prompt)
        print("=" * 50)
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        print("RESPONSE:")
        print(response.content)
        print("=" * 50)
        
        detailed_summary = response.content
        
        # 根据工作模式添加不同的结束语
        if self.workflow_mode == "structured":
            detailed_summary += PromptTemplates.STRUCTURED_MODE_COMPLETION_MESSAGE
        
        summary_message = AIMessage(content=detailed_summary)
        
        return {
            **state,
            "messages": state["messages"] + [summary_message],
            "summary": detailed_summary,
            "assessment_complete": True,
            "conversation_mode": "assessment_complete",
            "chat_therapist_active": True
        }
    
    def fallback_response(self, current_state: dict, user_message: str) -> tuple:
        """后备响应方法"""
        try:
            fallback_prompt = PromptTemplates.get_fallback_prompt(user_message)
            system_content = PromptTemplates.get_fallback_response_system_content(
                current_state.get("current_question_id", "")
            )
            
            response = self.llm.invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=fallback_prompt)
            ])
            
            return response.content, current_state
            
        except Exception as e:
            print(f"Fallback response error: {e}")
            return "抱歉，我现在无法理解您的意思。让我们继续我们的对话吧。", current_state