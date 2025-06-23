"""
增强的意图检测模块
提供更智能的对话模式检测和用户意图分析功能
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from agent_types import InterviewState


class EnhancedIntentDetector:
    """增强的意图检测器"""
    
    def __init__(self, llm, workflow_mode: str = "adaptive"):
        self.llm = llm
        self.workflow_mode = workflow_mode
        
        # 症状关键词库 - 分级管理
        self.symptom_keywords = {
            "high_severity": [
                "自杀", "自残", "想死", "轻生", "伤害自己", "不想活", "结束生命",
                "严重抑郁", "无法控制", "完全绝望", "崩溃", "精神崩溃"
            ],
            "medium_severity": [
                "抑郁", "焦虑", "恐慌", "失眠", "噩梦", "幻觉", "妄想",
                "强迫", "创伤", "PTSD", "双相", "躁郁", "精神分裂"
            ],
            "low_severity": [
                "情绪低落", "心情不好", "压力大", "紧张", "担心", "忧虑",
                "睡眠不好", "食欲不振", "注意力不集中", "疲劳", "烦躁"
            ]
        }
        
        # 闲聊意图关键词
        self.chat_keywords = [
            "闲聊", "聊天", "谈心", "聊聊", "随便聊", "陪我聊", "说说话",
            "聊天吧", "想聊天", "无聊", "陪陪我", "找人说话"
        ]
        
        # 问诊意图关键词
        self.interview_keywords = [
            "评估", "检查", "问诊", "测试", "诊断", "咨询", "了解我的情况",
            "心理健康", "精神状态", "心理状况", "开始评估"
        ]
        
        # 情绪状态检测模式
        self.emotion_patterns = {
            "distressed": ["痛苦", "难受", "煎熬", "崩溃", "绝望", "无助"],
            "anxious": ["紧张", "担心", "恐惧", "害怕", "焦虑", "不安"],
            "sad": ["伤心", "难过", "沮丧", "失落", "悲伤", "郁闷"],
            "angry": ["愤怒", "生气", "烦躁", "气愤", "恼火", "火大"],
            "confused": ["困惑", "迷茫", "不知道", "搞不清", "混乱", "糊涂"],
            "hopeful": ["希望", "期待", "想要", "希望能", "盼望", "憧憬"],
            "resistant": ["不想", "拒绝", "不愿意", "算了", "没必要", "不用了"]
        }
    
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
        detection_prompt = f"""
        作为专业的心理健康对话分析师，请分析用户的意图和需求：

        **当前对话信息：**
        - 用户输入："{latest_message}"
        - 对话轮数：第{current_turn + 1}轮
        - 检测到的情绪状态：{emotional_state}
        - 症状严重程度：{symptom_severity}
        - 最近对话历史：{conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history}

        **分析维度：**
        1. **意图分析**：用户是想要心理评估、症状问诊，还是寻求情感支持、闲聊陪伴？
        2. **紧急程度**：用户的状态是否需要立即关注和专业干预？
        3. **参与意愿**：用户对不同类型的对话（问诊vs闲聊）的开放程度如何？
        4. **情绪需求**：用户当前最需要的是专业评估还是情感支持？

        **判断标准：**
        - "interview"：用户提及具体症状、寻求评估、想了解自己的心理状况
        - "continue_interview"：用户正在回答问诊问题或继续描述症状
        - "chat"：用户明确想要闲聊、寻求陪伴、或者情绪低落需要支持
        - "supportive_chat"：用户情绪困扰但不需要正式评估，需要温暖陪伴

        请以JSON格式回复：
        {{
            "primary_intent": "interview/continue_interview/chat/supportive_chat",
            "confidence": 0.0-1.0,
            "reasoning": "详细的分析理由",
            "key_indicators": ["关键指标1", "关键指标2"],
            "emotional_needs": "用户的情感需求分析",
            "urgency_level": "low/medium/high",
            "recommended_approach": "建议的对话方式",
            "alternative_intent": "次要意图（如果有）"
        }}
        """
        
        try:
            print("🧠 增强意图检测 - LLM分析中...")
            
            detection_response = self.llm.invoke([
                SystemMessage(content="你是专业的心理健康对话分析师，具备深度分析用户意图和情感需求的能力。"),
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
        elif any(keyword in message for keyword in self.chat_keywords):
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
            
        # 对于低置信度的判断，给用户更多选择权
        if confidence < 0.6 and current_turn < 3:
            final_mode = "flexible_start"  # 新增的灵活开始模式
            
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