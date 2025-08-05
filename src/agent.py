"""
统一的Agent模块
整合了所有Agent相关的类型定义、核心功能和具体实现
"""

from typing import Dict, List, Optional, TypedDict, Annotated, Tuple
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages

from config import config
from prompts import PromptTemplates


# =========================
# 类型定义
# =========================

class InterviewState(TypedDict):
    """
    问诊状态 - 包含完整的会话状态信息
    
    基本状态字段：
    - messages: 对话消息列表
    - current_question_id: 当前问题ID
    - user_responses: 用户回答记录
    - assessment_complete: 评估是否完成
    - emergency_situation: 是否有紧急情况
    - summary: 评估总结
    - needs_followup: 是否需要追问
    - conversation_history: 对话历史记录
    - chief_complaint: 用户主要诉求
    - conversation_mode: 对话模式 ('idle', 'interview', 'chat', 'assessment_complete')
    - chat_therapist_active: CBT疗愈师是否激活
    - mode_detection_result: 模式检测结果
    - conversation_turn_count: 会话轮数计数器
    - interview_mode_locked: 问诊模式是否已锁定
    
    追问和分析字段：
    - followup_questions: 追问问题列表
    - risk_level: 风险等级 (low/medium/high)
    - risk_indicators: 风险指标列表
    - emotional_state: 用户情感状态描述
    - content_complete: 用户回答是否完整
    - understanding_summary: 对用户回答的理解总结
    - empathetic_response: 共情回应内容
    - current_analysis: 当前完整分析结果
    
    会话追踪字段：
    - debug_info: Debug分析信息列表
    - session_start_time: 会话开始时间
    - question_sequence: 问题序列记录
    - final_response: 最终回复内容
    - assessment_duration: 评估持续时间
    
    用户特征字段：
    - user_engagement_level: 用户参与度 (high/medium/low)
    - response_detail_level: 回答详细程度 (detailed/moderate/brief)
    
    临床评估字段：
    - symptoms_identified: 识别出的症状列表
    - domains_assessed: 已评估的领域列表
    - severity_indicators: 各领域严重程度指标
    """
    messages: Annotated[List[AnyMessage], add_messages]
    current_question_id: Optional[str]
    user_responses: Dict[str, str]
    assessment_complete: bool
    emergency_situation: bool
    summary: str
    needs_followup: bool
    conversation_history: List[str]
    chief_complaint: str  # 用户的主诉
    
    # 对话模式相关字段
    conversation_mode: str  # 对话模式：'idle', 'interview', 'chat', 'assessment_complete'
    chat_therapist_active: bool  # CBT疗愈师是否激活
    mode_detection_result: Dict  # 模式检测结果和分析
    conversation_turn_count: int  # 会话轮数计数器
    interview_mode_locked: bool  # 问诊模式是否已锁定
    
    # 追问相关字段
    followup_questions: List[str]  # 追问问题列表
    
    # 风险评估字段
    risk_level: str  # 风险等级：low/medium/high
    risk_indicators: List[str]  # 风险指标列表
    
    # 情感和理解分析字段
    emotional_state: str  # 用户的情感状态描述
    content_complete: bool  # 用户回答是否完整
    understanding_summary: str  # 对用户回答的理解总结
    empathetic_response: str  # 共情回应内容
    
    # 当前分析结果
    current_analysis: Dict  # 存储最近一次的完整分析结果
    
    # Debug和追踪信息
    debug_info: List[Dict]  # Debug分析信息列表
    session_start_time: str  # 会话开始时间
    question_sequence: List[str]  # 问题序列记录
    
    # 最终评估相关
    final_response: str  # 最终回复内容
    assessment_duration: int  # 评估持续时间（分钟）
    
    # 用户特征标记
    user_engagement_level: str  # 用户参与度：high/medium/low
    response_detail_level: str  # 回答详细程度：detailed/moderate/brief
    
    # 临床相关标记
    symptoms_identified: List[str]  # 识别出的症状列表
    domains_assessed: List[str]  # 已评估的领域列表
    severity_indicators: Dict[str, str]  # 各领域严重程度指标
    
    # 诊断标准追踪字段
    assessed_criteria: Dict[str, List[str]]  # 已评估的诊断标准，按障碍类型分组
    criteria_results: Dict[str, bool]  # 各个诊断标准的评估结果
    current_disorder_focus: str  # 当前关注的障碍类型
    
    # 新增状态字段
    current_topic: str  # 当前话题
    is_follow_up: bool  # 是否是追问状态


# =========================
# 核心Agent类
# =========================

class SCID5AgentCore:
    """SCID-5问诊代理核心类"""
    
    def __init__(self, workflow_mode: str = "adaptive"):
        """
        初始化问诊代理
        
        Args:
            workflow_mode: 工作流程模式
                - "adaptive": 智能检测模式，根据用户意图自由切换问诊和闲聊
                - "structured": 固定流程模式，先完成问诊再转CBT闲聊
        """
        self.workflow_mode = workflow_mode
        self.llm = self._initialize_llm()
        self.workflow = None
        self.app = None
    
    def _initialize_llm(self) -> ChatOpenAI:
        """初始化语言模型"""
        return ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE,
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )


# =========================
# 主要Agent实现
# =========================

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
        
        # 延迟导入以避免循环依赖
        from conversation_modes import ConversationModeHandler
        from interview_flow import InterviewFlowHandler  
        from response_generation import ResponseGenerator
        from emergency_handling import EmergencyHandler
        from workflow_builder import WorkflowBuilder
        
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
                
                if mode == "chat" or mode == "supportive_chat" or mode == "assessment_complete":
                    # 使用CBT疗愈师回应
                    print(f"🎯 进入CBT疗愈师模式，检测模式: {mode}", flush=True)
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
            print(f"DEBUG: process_message_sync error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return error_msg, state or {}
    
    def _get_next_question_response(self, current_question_id: str, user_response: str, state: dict) -> str:
        """根据当前问题和用户回答生成下一个问题"""
        return self.response_generator.get_next_question_response(current_question_id, user_response, state)


# =========================
# 工厂函数
# =========================

def create_agent(workflow_mode: str = "adaptive") -> SCID5Agent:
    """
    创建SCID5代理实例
    
    Args:
        workflow_mode: 工作流程模式
            - "adaptive": 智能检测模式，根据用户意图自由切换问诊和闲聊
            - "structured": 固定流程模式，先完成问诊再转CBT闲聊
    
    Returns:
        SCID5Agent: 配置好的代理实例
    """
    return SCID5Agent(workflow_mode=workflow_mode)


# 全局代理实例
# 可以通过修改这里的workflow_mode参数来切换工作模式：
# - "adaptive": 智能检测模式，根据用户意图自由切换问诊和闲聊
# - "structured": 固定流程模式，先完成问诊再转CBT闲聊

# 延迟初始化全局实例以避免循环导入
_global_agent = None

def get_scid5_agent(workflow_mode: str = "adaptive") -> SCID5Agent:
    """获取全局SCID5代理实例（延迟初始化）"""
    global _global_agent
    if _global_agent is None:
        _global_agent = SCID5Agent(workflow_mode=workflow_mode)
    return _global_agent

# 为了向后兼容，可以这样使用：
# scid5_agent = get_scid5_agent()