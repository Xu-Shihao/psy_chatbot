"""
增强的意图检测模块
提供更智能的对话模式检测和用户意图分析功能
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from agent import InterviewState
from prompts import PromptTemplates, KeywordLibrary


class EnhancedIntentDetector:
    """增强的意图检测器"""
    
    def __init__(self, openai_client, workflow_mode: str = "adaptive"):
        self.openai_client = openai_client
        self.workflow_mode = workflow_mode
        
        # 使用统一的关键词库
        self.symptom_keywords = KeywordLibrary.SYMPTOM_KEYWORDS
        self.chat_keywords = KeywordLibrary.CHAT_KEYWORDS
        self.interview_keywords = KeywordLibrary.INTERVIEW_KEYWORDS
        self.emotion_patterns = KeywordLibrary.EMOTION_PATTERNS
    
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
        for emotion, keywords in self.emotion_patterns.items():
            if any(keyword in message for keyword in keywords):
                return emotion
        return "neutral"
    
    def _assess_symptom_severity(self, message: str) -> str:
        """评估症状严重程度"""
        for severity, keywords in self.symptom_keywords.items():
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
        for keyword in self.symptom_keywords["high_severity"]:
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
            
            # 使用OpenAI API直接调用
            from agent import call_openai_api
            openai_messages = [
                {"role": "system", "content": PromptTemplates.INTENT_ANALYST_SYSTEM_PROMPT},
                {"role": "user", "content": detection_prompt}
            ]
            detection_response_content = call_openai_api(self.openai_client, openai_messages)
            
            result = json.loads(detection_response_content)
            print(f"✅ 检测结果：{result['primary_intent']} (置信度: {result['confidence']})")
            
            return result
            
        except Exception as e:
            print(f"❌ 增强检测失败，使用后备逻辑: {e}")
            return self._fallback_detection(latest_message, context)
    
    def _fallback_detection(self, message: str, context: Dict) -> Dict:
        """后备检测逻辑"""
        symptom_severity = context["symptom_severity"]
        emotional_state = context["emotional_state"]
        current_turn = context["current_turn"]
        
        # 明确表达问诊需求的关键词检测
        if any(keyword in message for keyword in KeywordLibrary.INTERVIEW_KEYWORDS):
            return {
                "primary_intent": "interview",
                "confidence": 0.9,
                "reasoning": "用户明确表达问诊需求",
                "urgency_level": "high"
            }
        
        # 高严重度症状仍然触发问诊（紧急情况）
        if symptom_severity == "high":
            return {
                "primary_intent": "interview",
                "confidence": 0.8,
                "reasoning": f"检测到{symptom_severity}级别症状，建议问诊评估",
                "urgency_level": "high"
            }
        
        # 明确表达闲聊需求
        if any(keyword in message for keyword in KeywordLibrary.CHAT_KEYWORDS):
            return {
                "primary_intent": "chat",
                "confidence": 0.8,
                "reasoning": "用户明确表达想要闲聊",
                "urgency_level": "low"
            }
        
        # 前3轮：默认为支持性对话，除非明确要求问诊
        if current_turn < 3:
            return {
                "primary_intent": "chat",
                "confidence": 0.6,
                "reasoning": "前3轮对话，默认进入闲聊模式",
                "urgency_level": "low"
            }
        
        # 3轮后：基于情绪状态和症状判断
        if symptom_severity == "medium":
            return {
                "primary_intent": "supportive_chat",
                "confidence": 0.7,
                "reasoning": f"检测到{symptom_severity}级别症状，提供支持性对话",
                "urgency_level": "medium"
            }
        elif emotional_state in ["distressed", "anxious", "sad"]:
            return {
                "primary_intent": "supportive_chat",
                "confidence": 0.7,
                "reasoning": f"用户情绪状态为{emotional_state}，需要支持性对话",
                "urgency_level": "medium"
            }
        else:
            return {
                "primary_intent": "chat",
                "confidence": 0.5,
                "reasoning": "无明确意图，默认闲聊模式",
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
            
        # 前3轮对话：没有明确问诊意图时，默认切换到CBT闲聊模式
        if current_turn < 3:
            # 只有明确的高置信度问诊意图才切换到问诊模式
            if primary_intent in ["interview", "continue_interview"] and confidence > 0.8:
                final_mode = primary_intent
                chat_active = False
                should_lock = True
                print(f"🔒 前3轮检测到明确问诊意图（置信度: {confidence:.2f}），切换到问诊模式", flush=True)
            else:
                final_mode = "chat"  # 前3轮默认进入CBT闲聊模式
                chat_active = True
                should_lock = False
                print(f"🔄 前3轮对话，默认进入CBT闲聊模式（置信度: {confidence:.2f}）", flush=True)
        # 3轮后：对于低置信度的判断，仍然默认切换到CBT闲聊模式
        elif confidence < 0.6:
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