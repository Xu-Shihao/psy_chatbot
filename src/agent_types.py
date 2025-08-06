"""
类型定义模块
包含问诊系统中使用的所有类型定义
"""

from typing import Dict, List, Optional, TypedDict, Annotated
from langgraph.graph.message import AnyMessage, add_messages

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
    conversation_history: List[Dict[str, str]]  # OpenAI格式: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
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