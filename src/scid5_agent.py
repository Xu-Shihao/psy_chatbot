"""
SCID-5问诊代理主类
整合所有功能模块，提供完整的问诊功能
"""

from typing import Optional, Dict, Tuple
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from agent_types import InterviewState
from agent_core import SCID5AgentCore
from conversation_modes import ConversationModeHandler
from interview_flow import InterviewFlowHandler
from response_generation import ResponseGenerator
from emergency_handling import EmergencyHandler
from workflow_builder import WorkflowBuilder
from scid5_knowledge import scid5_kb


class SCID5Agent(SCID5AgentCore):
    """SCID-5问诊代理"""
    
    def __init__(self, workflow_mode: str = "adaptive"):
        """
        初始化问诊代理
        
        Args:
            workflow_mode: 工作流程模式
                - "adaptive": 智能检测模式，根据用户意图自由切换问诊和闲聊
                - "structured": 固定流程模式，先完成问诊再转CBT闲聊
        """
        super().__init__(workflow_mode)
        
        # 初始化各个功能模块
        self.conversation_handler = ConversationModeHandler(self.llm, workflow_mode)
        self.interview_handler = InterviewFlowHandler(self.llm)
        self.response_generator = ResponseGenerator(self.llm)
        self.emergency_handler = EmergencyHandler(self.llm)
        
        # 为响应生成器设置工作模式
        self.response_generator.workflow_mode = workflow_mode
        
        # 创建工作流
        self.workflow_builder = WorkflowBuilder(self)
        self.workflow = self.workflow_builder.create_workflow()
        self.app = self.workflow.compile()
    
    # 代理所有方法到相应的处理器
    def start_interview(self, state: InterviewState) -> InterviewState:
        """开始问诊"""
        return self.interview_handler.start_interview(state)
    
    def detect_conversation_mode(self, state: InterviewState) -> InterviewState:
        """检测对话模式"""
        return self.conversation_handler.detect_conversation_mode(state)
    
    def chat_therapist_response(self, state: InterviewState) -> InterviewState:
        """CBT疗愈师响应"""
        return self.conversation_handler.chat_therapist_response(state)
    
    def understand_and_respond(self, state: InterviewState) -> InterviewState:
        """理解用户回答并提供共情回应"""
        return self.interview_handler.understand_and_respond(state)
    
    def ask_question(self, state: InterviewState) -> InterviewState:
        """智能提问"""
        return self.interview_handler.ask_question(state)
    
    def check_emergency(self, state: InterviewState) -> InterviewState:
        """检查紧急情况"""
        return self.emergency_handler.check_emergency(state)
    
    def generate_summary(self, state: InterviewState) -> InterviewState:
        """生成评估总结"""
        return self.response_generator.generate_summary(state)
    
    def emergency_response(self, state: InterviewState) -> InterviewState:
        """处理紧急情况"""
        return self.emergency_handler.emergency_response(state)
    
    # 工作流控制方法
    def should_continue_after_mode_detection(self, state: InterviewState) -> str:
        """决定模式检测后的下一步"""
        return self.workflow_builder.should_continue_after_mode_detection(state)
    
    def should_continue_after_understand_and_respond(self, state: InterviewState) -> str:
        """理解和回应后的流程控制"""
        return self.workflow_builder.should_continue_after_understand_and_respond(state)
    
    def should_continue_after_question(self, state: InterviewState) -> str:
        """问题后的流程控制（保留用于兼容性）"""
        return self.workflow_builder.should_continue_after_question(state)
    
    def should_continue_after_check(self, state: InterviewState) -> str:
        """检查后的流程控制"""
        return self.workflow_builder.should_continue_after_check(state)
    
    async def process_message(self, user_message: str, state: Optional[InterviewState] = None) -> Tuple[str, InterviewState]:
        """处理用户消息"""
        if state is None:
            # 初次对话，开始问诊流程
            result = await self.app.ainvoke({
                "messages": [],
                "current_question_id": None,
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
                "question_sequence": [],
                "final_response": "",
                "assessment_duration": 0,
                "user_engagement_level": "medium",
                "response_detail_level": "moderate",
                "symptoms_identified": [],
                "domains_assessed": [],
                "severity_indicators": {},
                "chief_complaint": "",
                "conversation_mode": "interview",
                "chat_therapist_active": False,
                "mode_detection_result": {},
                "current_topic": "",
                "is_follow_up": False
            })
        else:
            # 继续对话，添加用户消息并处理
            user_msg = HumanMessage(content=user_message)
            updated_state = {
                **state,
                "messages": state["messages"] + [user_msg]
            }
            
            # 先理解和回应用户的话
            analyzed_state = self.understand_and_respond(updated_state)
            
            # 然后继续工作流
            result = await self.app.ainvoke(analyzed_state)
        
        # 获取最后一条AI消息
        ai_response = ""
        if result["messages"]:
            for msg in reversed(result["messages"]):
                if isinstance(msg, (AIMessage, SystemMessage)):
                    ai_response = msg.content if hasattr(msg, 'content') else str(msg)
                    break
        
        return ai_response, result

    def process_message_sync(self, user_message: str, state: Optional[dict] = None) -> Tuple[str, dict]:
        """处理用户消息 - 同步版本，使用新的工作流程"""
        try:
            # 如果是开始评估或者没有状态，使用工作流初始化
            if user_message == "开始评估" or state is None:
                # 创建初始状态，使用start_interview方法
                initial_state = self.start_interview({
                    "messages": [],
                    "current_question_id": None,
                    "user_responses": {},
                    "assessment_complete": False,
                    "emergency_situation": False,
                    "summary": "",
                    "needs_followup": False,
                    "conversation_history": [],
                    "chief_complaint": "",
                    "conversation_mode": "idle",
                    "chat_therapist_active": False,
                    "mode_detection_result": {},
                    "conversation_turn_count": 0,
                    "interview_mode_locked": False,
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
                    "question_sequence": [],
                    "final_response": "",
                    "assessment_duration": 0,
                    "user_engagement_level": "medium",
                    "response_detail_level": "moderate",
                    "symptoms_identified": [],
                    "domains_assessed": [],
                    "severity_indicators": {},
                    "current_topic": "",
                    "is_follow_up": False
                })
                
                # 从messages中获取AI的自我介绍回复
                ai_response = ""
                if initial_state["messages"]:
                    for msg in reversed(initial_state["messages"]):
                        if isinstance(msg, AIMessage):
                            ai_response = msg.content if hasattr(msg, 'content') else str(msg)
                            break
                
                return ai_response, initial_state
            
            # 处理用户输入，使用新的工作流程
            if state:
                # 添加用户消息到状态
                current_state = state.copy()
                current_state["messages"] = current_state.get("messages", []) + [HumanMessage(content=user_message)]
                
                # 步骤1: 检测对话模式
                mode_state = self.detect_conversation_mode(current_state)
                
                # 根据检测到的模式选择处理路径
                mode = mode_state.get("conversation_mode", "interview")
                
                if mode == "chat" or mode == "assessment_complete":
                    # 使用CBT疗愈师回应
                    result_state = self.chat_therapist_response(mode_state)
                    ai_response = result_state.get("final_response", "我很高兴和您聊天，有什么想聊的吗？")
                    return ai_response, result_state
                
                elif mode == "interview" or mode == "continue_interview":
                    # 使用问诊流程
                    # 理解和回应用户输入（已包含下一个问题的生成）
                    understood_state = self.understand_and_respond(mode_state)
                    
                    # 从messages中获取回应
                    ai_response = ""
                    if understood_state["messages"]:
                        for msg in reversed(understood_state["messages"]):
                            if isinstance(msg, AIMessage):
                                ai_response = msg.content
                                break
                    
                    # 检查是否需要特殊处理
                    if understood_state.get("emergency_situation", False):
                        # 处理紧急情况
                        emergency_state = self.emergency_response(understood_state)
                        # 从messages中获取紧急响应
                        emergency_response = ""
                        if emergency_state["messages"]:
                            for msg in reversed(emergency_state["messages"]):
                                if isinstance(msg, AIMessage):
                                    emergency_response = msg.content
                                    break
                        return emergency_response, emergency_state
                    
                    elif understood_state.get("assessment_complete", False):
                        # 生成总结
                        summary_state = self.generate_summary(understood_state)
                        # 从messages中获取总结
                        summary_response = ""
                        if summary_state["messages"]:
                            for msg in reversed(summary_state["messages"]):
                                if isinstance(msg, AIMessage):
                                    summary_response = msg.content
                                    break
                        return summary_response, summary_state
                    
                    else:
                        # 返回理解和回应的结果（已包含下一个问题）
                        return ai_response, understood_state
                
                else:
                    # 默认使用问诊模式
                    return self.response_generator.fallback_response(current_state, user_message)
            
            return "请重新开始评估。", {}
            
        except Exception as e:
            error_msg = f"抱歉，处理您的消息时出现错误：{str(e)}"
            print(f"DEBUG: process_message_sync error: {e}")
            import traceback
            traceback.print_exc()
            return error_msg, state or {}
    
    def _get_next_question_response(self, current_question_id: str, user_response: str, state: dict) -> str:
        """根据当前问题和用户回答生成下一个问题"""
        return self.response_generator.get_next_question_response(current_question_id, user_response, state)