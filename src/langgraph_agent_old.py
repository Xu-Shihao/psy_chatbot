"""
基于LangGraph的SCID-5问诊代理
实现结构化的精神疾病问诊流程
"""

from typing import Dict, List, Optional, TypedDict, Annotated
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages

from config import config
from scid5_knowledge import scid5_kb, Question

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
    
    # 新增状态字段
    current_topic: str  # 当前话题
    is_follow_up: bool  # 是否是追问状态

class SCID5Agent:
    """SCID-5问诊代理"""
    
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
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()
    
    def _initialize_llm(self) -> ChatOpenAI:
        """初始化语言模型"""
        return ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_API_BASE,
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )
    
    def _create_workflow(self) -> StateGraph:
        """创建问诊工作流"""
        workflow = StateGraph(InterviewState)
        
        # 添加节点
        workflow.add_node("start_interview", self.start_interview)
        workflow.add_node("detect_conversation_mode", self.detect_conversation_mode)
        workflow.add_node("chat_therapist_response", self.chat_therapist_response)
        workflow.add_node("understand_and_respond", self.understand_and_respond)
        workflow.add_node("ask_question", self.ask_question)
        workflow.add_node("check_emergency", self.check_emergency)
        workflow.add_node("generate_summary", self.generate_summary)
        workflow.add_node("emergency_response", self.emergency_response)
        
        # 设置入口点
        workflow.set_entry_point("start_interview")
        
        # 定义边
        workflow.add_edge("start_interview", "detect_conversation_mode")
        workflow.add_conditional_edges(
            "detect_conversation_mode",
            self.should_continue_after_mode_detection,
            {
                "interview": "ask_question",
                "chat": "chat_therapist_response",
                "continue_interview": "understand_and_respond",
                "assessment_complete": "chat_therapist_response"
            }
        )
        workflow.add_edge("chat_therapist_response", END)
        workflow.add_conditional_edges(
            "ask_question",
            self.should_continue_after_question,
            {
                "wait_response": END,
                "emergency": "emergency_response"
            }
        )
        workflow.add_edge("understand_and_respond", "check_emergency")
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
    
    def start_interview(self, state: InterviewState) -> InterviewState:
        """开始问诊 - 生成自我介绍和引导开场白"""
        system_message = SystemMessage(content="""
        你的名字叫灵溪智伴，是一个心理咨询师，能根据用户的回答来进行预问诊和心理疏导。

        请遵循以下原则：
        1. 保持专业、同理心和非批判的态度
        2. 根据用户的情感状态提供足够的共情回应
        3. 基于用户回答进行适当的深入询问
        4. 保持对话的自然对话的流畅性，不要过于机械化的提问，要像一个心理咨询师一样与用户对话
        5. 如发现紧急情况（如自杀风险），立即提供危机干预信息
        6. 提醒这是筛查工具，不能替代专业诊断

        对话风格要求：
        - 温暖、理解、非批判
        - 适时表达关切和理解
        - 鼓励用户分享更多细节
        - 根据用户的情绪状态调整语气
        - 以灵溪智伴的身份与用户建立专业而温暖的关系
        """)
        
        # 生成自我介绍和引导开场白
        try:
            initial_prompt = """
你的名字叫灵溪智伴，是一位温暖、专业的心理咨询师，正在与一位新的来访者开始第一次会面。

请以灵溪智伴的身份生成一个自我介绍和开场白，要求：
1. 温暖地介绍自己的姓名和身份（心理咨询师）
2. 表达关心和欢迎来访者的感受
3. 营造安全、无批判的咨询环境氛围
4. 引导用户分享他们目前遇到的问题或想要谈论的事情
5. 询问用户的主诉（最主要的困扰或希望解决的问题）
6. 鼓励用户真实、详细地表达自己的感受和经历
7. 语气要专业而温暖，像一位有经验的心理咨询师
8. 让用户感到被理解和被接纳

请直接给出自我介绍和开场白，控制在250字以内。
"""
            
            print("=" * 50)
            print("🔍 DEBUG - START_INTERVIEW LLM CALL")
            print("PROMPT:")
            print(initial_prompt)
            print("=" * 50)
            
            response = self.llm.invoke([HumanMessage(content=initial_prompt)])
            initial_response = response.content.strip()
            
            print("RESPONSE:")
            print(initial_response)
            print("=" * 50)
            
            # 添加重要说明
            initial_response += """

**重要说明：**
- 这里是一个安全、保密的空间，请放心分享您的真实感受
- 没有对错的答案，我在这里是为了理解和帮助您
- 这是一个心理健康筛查工具，不能替代专业医疗诊断  
- 如有任何紧急情况，请立即寻求专业帮助
- 我们的对话完全保密"""
            
        except Exception as e:
            print(f"🔍 DEBUG: LLM生成开头失败: {e}")
            # 如果LLM调用失败，使用备用开头
            initial_response = """您好！我是灵溪智伴，一位心理咨询师。很高兴在这里与您相遇 😊

首先，我想让您知道这里是一个安全、无批判的空间。我在这里是为了倾听和理解您，陪伴您一起面对您正在经历的困扰。

我想请您跟我分享一下，是什么让您今天来到这里？您目前最主要的困扰或希望解决的问题是什么？请尽可能详细地告诉我您的感受和经历，不用担心说得不够好或不够准确。

**重要说明：**
- 这里是一个安全、保密的空间，请放心分享您的真实感受
- 没有对错的答案，我在这里是为了理解和帮助您
- 这是一个心理健康筛查工具，不能替代专业医疗诊断
- 如有任何紧急情况，请立即寻求专业帮助
- 我们的对话完全保密"""
        
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
            "is_follow_up": False
        }
    
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
    
    def understand_and_respond(self, state: InterviewState) -> InterviewState:
        """理解用户回答并提供共情回应"""
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
        
        # 构建理解和回应的提示
        conversation_context = "\n".join(state["conversation_history"][-3:]) if state["conversation_history"] else ""
        
        understanding_prompt = f"""
基于以下对话情境，理解用户的回答并提供适当的回应：

当前问题背景：{current_question.text if current_question else "无"}

最近对话历史：{conversation_context}

用户回答：{last_user_message}

请进行以下分析和回应：

1. **情感理解**：识别用户回答中的情感状态（如焦虑、悲伤、恐惧、无助等）

2. **内容分析**：
    - 用户是否完整回答了问题？
    - 是否需要进一步澄清或深入了解？
    - 是否有重要的细节需要追问？

3. **共情回应**：基于用户的情感状态，提供温暖、理解的回应

4. **追问建议**：如果需要，提出1-2个自然的追问问题

5. **风险评估**：是否涉及自伤、自杀或其他紧急情况？

请以JSON格式回复：
{{
    "emotional_state": "用户的情感状态描述",
    "content_complete": true/false,
    "empathetic_response": "共情回应内容",
    "followup_needed": true/false,
    "followup_questions": ["追问问题1", "追问问题2"],
    "risk_level": "low/medium/high",
    "understanding_summary": "对用户回答的理解总结"
}}
        """
        
        print("=" * 50)
        print("🔍 DEBUG - UNDERSTAND_AND_RESPOND LLM CALL")
        print("PROMPT:")
        print(understanding_prompt)
        print("=" * 50)
        
        response = self.llm.invoke([HumanMessage(content=understanding_prompt)])
        
        print("RESPONSE:")
        print(response.content)
        print("=" * 50)
        
        try:
            analysis = json.loads(response.content)
        except:
            # 如果JSON解析失败，提供默认回应
            analysis = {
                "emotional_state": "未能识别",
                "content_complete": True,
                "empathetic_response": "谢谢您的分享。",
                "followup_needed": False,
                "followup_questions": [],
                "risk_level": "low",
                "risk_indicators": [],
                "understanding_summary": last_user_message
            }
        
        # 记录用户回答
        updated_responses = state["user_responses"].copy()
        if current_question_id:
            updated_responses[current_question_id] = last_user_message
        
        # 更新对话历史
        updated_history = state["conversation_history"].copy()
        updated_history.append(f"Q: {current_question.text if current_question else ''}")
        updated_history.append(f"A: {last_user_message}")
        # updated_history.append(f"Analysis: {analysis['understanding_summary']}")
        
        # 生成回应消息
        response_parts = []
        
        # 添加共情回应
        if analysis["empathetic_response"]:
            response_parts.append(analysis["empathetic_response"])
        
        # 添加追问
        if analysis["followup_needed"] and analysis["followup_questions"]:
            response_parts.append("\n我想进一步了解一下：")
            for i, question in enumerate(analysis["followup_questions"][:2], 1):
                response_parts.append(f"{i}. {question}")
        
        if response_parts:
            ai_response = AIMessage(content="\n".join(response_parts))
            messages_to_add = [ai_response]
        else:
            messages_to_add = []
        
        return {
            **state,
            "messages": state["messages"] + messages_to_add,
            "user_responses": updated_responses,
            "needs_followup": analysis["followup_needed"],
            "conversation_history": updated_history,
            "emergency_situation": analysis["risk_level"] == "high",
            "risk_level": analysis["risk_level"],
            "risk_indicators": analysis.get("risk_indicators", []),
            "emotional_state": analysis["emotional_state"],
            "content_complete": analysis["content_complete"],
            "understanding_summary": analysis["understanding_summary"],
            "empathetic_response": analysis["empathetic_response"],
            "current_analysis": analysis,
            "session_start_time": state["session_start_time"],
            "question_sequence": state["question_sequence"] + [current_question_id] if current_question_id else state["question_sequence"],
            "assessment_duration": state["assessment_duration"] + 1 if current_question_id else state["assessment_duration"],
            "user_engagement_level": "high" if analysis["risk_level"] == "high" else "medium" if analysis["risk_level"] == "medium" else "low",
            "response_detail_level": "detailed" if len(analysis.get("risk_indicators", [])) > 0 else "moderate" if analysis["risk_level"] == "medium" else "brief",
            "symptoms_identified": state["symptoms_identified"] + [current_question.text] if current_question else state["symptoms_identified"],
            "domains_assessed": state["domains_assessed"] + [current_question.category.value] if current_question else state["domains_assessed"],
            "severity_indicators": {**state["severity_indicators"], **{current_question.category.value: analysis["risk_level"]}} if current_question else state["severity_indicators"],
            "chief_complaint": state["chief_complaint"],
            "conversation_mode": state["conversation_mode"],
            "chat_therapist_active": state["chat_therapist_active"],
            "mode_detection_result": state["mode_detection_result"],
            "current_topic": state["current_topic"],
            "is_follow_up": state["is_follow_up"]
        }
    
    def ask_question(self, state: InterviewState) -> InterviewState:
        """智能提问"""
        current_question_id = state["current_question_id"]
        
        # 如果需要追问，不要问新问题
        if state.get("needs_followup", False):
            return state
        
        if not current_question_id:
            return {**state, "assessment_complete": True}
        
        current_question = scid5_kb.questions.get(current_question_id)
        if not current_question:
            return {**state, "assessment_complete": True}
        
        # 获取对话上下文
        conversation_context = "\n".join(state["conversation_history"][-10:]) if state["conversation_history"] else ""
        
        # 使用LLM生成更智能、更自然的问题
        question_prompt = f"""
基于以下信息，以温暖、专业的方式提出下一个问题：

结构化问题：{current_question.text}
问题目的：{current_question.purpose if hasattr(current_question, 'purpose') else '评估相关症状'}

对话历史：
{conversation_context}

请遵循以下要求：
1. 保持问题的诊断目的，但使用更自然、温和的口语表达方式
2. 根据对话历史，适当调整问题的表述和语气
3. 如果用户之前表现出情绪困扰，在问题前加入适当的关怀表达
4. 使问题听起来像自然对话，而非机械化的清单检查，不要使用1.2.3.这样的格式
5. 如果需要，可以提供一些解释或背景信息帮助用户理解
6. 如果用户前序回答中已经说没有自杀想法，不要持续询问自杀和自伤的问题

直接返回优化后的问题文本，不需要其他格式。
        """
        
        print("=" * 50)
        print("🔍 DEBUG - ASK_QUESTION LLM CALL")
        print("PROMPT:")
        print(question_prompt)
        print("=" * 50)
        
        response = self.llm.invoke([HumanMessage(content=question_prompt)])
        
        print("RESPONSE:")
        print(response.content)
        print("=" * 50)
        
        formatted_question = response.content.strip()
        
        # 添加问题到消息中
        question_message = AIMessage(content=formatted_question)
        
        return {
            **state,
            "messages": state["messages"] + [question_message],
            "needs_followup": False
        }
    
    def check_emergency(self, state: InterviewState) -> InterviewState:
        """检查紧急情况"""
        # 获取最近的用户回答
        last_response = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_response = msg.content.lower()
                break
        
        # 使用LLM进行更智能的风险评估
        risk_prompt = f"""请评估以下用户回答中的自杀风险或紧急心理健康风险：

用户回答：{last_response}
当前问题ID：{state.get("current_question_id", "")}

评估标准：
1. 直接或间接提到自杀想法
2. 表达绝望、无助感
3. 提到自伤行为
4. 表达想要"消失"或"结束一切"
5. 提到具体的自杀计划或方法

请以JSON格式回复：
{{
    "risk_level": "low/medium/high",
    "risk_indicators": ["识别到的风险指标列表"],
    "immediate_action_needed": true/false,
    "reasoning": "风险评估理由"
}}
        """
        
        print("=" * 50)
        print("🔍 DEBUG - CHECK_EMERGENCY LLM CALL")
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
            emergency_keywords = ["自杀", "结束生命", "不想活", "死了算了", "自杀计划"]
            emergency_detected = any(keyword in last_response for keyword in emergency_keywords)
        
        return {
            **state,
            "emergency_situation": emergency_detected
        }
    
    def should_continue_after_mode_detection(self, state: InterviewState) -> str:
        """决定模式检测后的下一步"""
        # 如果是结构化模式且未完成评估，强制进入问诊流程
        if (self.workflow_mode == "structured" and 
            not state.get("assessment_complete", False)):
            return "interview"
        
        # 智能检测模式的原有逻辑
        mode = state.get("conversation_mode", "interview")
        
        if mode == "chat":
            return "chat"
        elif mode == "interview":
            return "interview"
        elif mode == "continue_interview":
            return "continue_interview"
        elif mode == "assessment_complete":
            return "assessment_complete"
        else:
            return "interview"  # 默认进入问诊模式
    
    def should_continue_after_question(self, state: InterviewState) -> str:
        """问题后的流程控制"""
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
                if isinstance(msg, HumanMessage):
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
    
    def generate_summary(self, state: InterviewState) -> InterviewState:
        """生成评估总结"""
        assessment_summary = scid5_kb.generate_assessment_summary()
        
        # 使用LLM生成更详细和个性化的总结
        prompt = f"""基于以下完整的心理健康筛查对话，生成一份专业、个性化的评估总结：

对话历史：
{chr(10).join(state["conversation_history"])}

用户回答记录：
{json.dumps(state["user_responses"], ensure_ascii=False, indent=2)}

结构化评估总结：
{assessment_summary}

请生成一份包含以下内容的个性化总结：

## 🔍 评估总结

## 评估结果
- 主要的障碍类型
- 可以是多个障碍类型
- 可以是只是有症状，没有明确诊断

## 现病史
- 基于对话内容总结用户的主要困扰和症状表现
- 识别的情感模式和行为特征

## 既往史
- 如果有既往史，请总结既往史

## 家族史
- 如果有家族史，请总结家族史

### 💡 建议和后续步骤
- 具体的下一步行动建议
- 可能有帮助的资源和支持

### ⚠️ 重要说明
- 强调这是筛查工具，不能替代专业诊断
- 鼓励寻求专业帮助

请保持温暖、支持性的语调，避免使用可能引起焦虑的医学术语。"""
        
        print("=" * 50)
        print("🔍 DEBUG - GENERATE_SUMMARY LLM CALL")
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
            detailed_summary += """
            
---

🌟 **问诊评估已完成！**

现在我们可以进入更轻松的交流环节。如果您有任何想聊的话题，或者需要情绪支持和心理建议，我很乐意以CBT疗愈师的身份继续陪伴您。

您可以：
- 分享您现在的感受
- 聊聊日常生活中的事情  
- 寻求应对困难的建议
- 或者任何您想谈论的话题

我在这里陪伴您 💕
"""
        
        summary_message = AIMessage(content=detailed_summary)
        
        return {
            **state,
            "messages": state["messages"] + [summary_message],
            "summary": detailed_summary,
            "assessment_complete": True,
            "conversation_mode": "assessment_complete",
            "chat_therapist_active": True
        }
    
    def emergency_response(self, state: InterviewState) -> InterviewState:
        """处理紧急情况"""
        emergency_message = AIMessage(content="""
        🚨 **我很担心您的安全**

        从您刚才的话中，我感受到您现在可能很痛苦。我想让您知道，您不是孤单的，而且寻求帮助是非常勇敢的行为。

        **请立即采取以下行动：**

        🏥 **立即寻求专业帮助：**
        - 前往最近的医院急诊科
        - 拨打以下紧急求助热线：
          - 全国心理援助热线：**400-161-9995**
          - 北京危机干预热线：**400-161-9995** 
          - 上海心理援助热线：**021-34289888**
          - 青少年心理热线：**12355**

        👥 **寻求身边的支持：**
        - 立即联系信任的朋友或家人
        - 请他们陪伴在您身边
        - 不要独自一人

        🛡️ **确保环境安全：**
        - 移开可能用于自伤的物品
        - 待在安全、有人的地方

        **请记住：**
        ✨ 您的生命很宝贵  
        ✨ 现在的痛苦是暂时的  
        ✨ 专业帮助是有效的  
        ✨ 有很多人关心您  

        我知道现在很困难，但请相信情况会好转的。专业的帮助能够支持您度过这个艰难时期。

        **您现在最重要的事情就是确保自己的安全。请立即寻求帮助。**
        """)
        
        return {
            **state,
            "messages": state["messages"] + [emergency_message],
            "assessment_complete": True
        }
    
    async def process_message(self, user_message: str, state: Optional[InterviewState] = None) -> tuple[str, InterviewState]:
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

    def process_message_sync(self, user_message: str, state: Optional[dict] = None) -> tuple[str, dict]:
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
                    # 步骤2: 理解和回应用户输入
                    understood_state = self.understand_and_respond(mode_state)
                    
                    # 步骤3: 检查紧急情况
                    checked_state = self.check_emergency(understood_state)
                    
                    if checked_state.get("emergency_situation", False):
                        # 处理紧急情况
                        emergency_state = self.emergency_response(checked_state)
                        # 从messages中获取紧急响应
                        ai_response = ""
                        if emergency_state["messages"]:
                            for msg in reversed(emergency_state["messages"]):
                                if isinstance(msg, AIMessage):
                                    ai_response = msg.content
                                    break
                        return ai_response, emergency_state
                    
                    elif checked_state.get("assessment_complete", False):
                        # 生成总结
                        summary_state = self.generate_summary(checked_state)
                        # 从messages中获取总结
                        ai_response = ""
                        if summary_state["messages"]:
                            for msg in reversed(summary_state["messages"]):
                                if isinstance(msg, AIMessage):
                                    ai_response = msg.content
                                    break
                        return ai_response, summary_state
                    
                    else:
                        # 继续提问
                        question_state = self.ask_question(checked_state)
                        # 从messages中获取问题
                        ai_response = ""
                        if question_state["messages"]:
                            for msg in reversed(question_state["messages"]):
                                if isinstance(msg, AIMessage):
                                    ai_response = msg.content
                                    break
                        return ai_response, question_state
                
                else:
                    # 默认使用问诊模式
                    return self._fallback_response(current_state, user_message)
            
            return "请重新开始评估。", {}
            
        except Exception as e:
            error_msg = f"抱歉，处理您的消息时出现错误：{str(e)}"
            print(f"DEBUG: process_message_sync error: {e}")
            import traceback
            traceback.print_exc()
            return error_msg, state or {}
    
    def _fallback_response(self, state: dict, user_message: str) -> tuple[str, dict]:
        """后备响应机制"""
        try:
            fallback_prompt = f"用户说：{user_message}"
            system_content = "你是灵溪智伴，一位专业的心理咨询师。"
            
            print("=" * 50)
            print("🔍 DEBUG - FALLBACK_RESPONSE LLM CALL")
            print("SYSTEM PROMPT:")
            print(system_content)
            print("USER PROMPT:")
            print(fallback_prompt)
            print("=" * 50)
            
            response = self.llm.invoke([
                SystemMessage(content=system_content),
                HumanMessage(content=fallback_prompt)
            ])
            
            print("RESPONSE:")
            print(response.content)
            print("=" * 50)
            
            state["final_response"] = response.content
            return response.content, state
            
        except Exception as e:
            fallback_msg = "抱歉，我现在无法很好地回应您。请您再试一次，或者告诉我您想要进行问诊还是想要闲聊。"
            return fallback_msg, state
    
    def _get_next_question_response(self, current_question_id: str, user_response: str, state: dict) -> str:
        """根据当前问题和用户回答生成下一个问题"""
        try:
            # 使用LLM生成个性化的回应和下一个问题
            conversation_context = "\n".join(state.get("conversation_history", []))[-500:]  # 限制长度
            
            # 根据debug模式调整prompt
            if config.DEBUG:
                prompt = f"""
你的名字叫马心宁，是一位专业的心理咨询师，正在进行SCID-5结构化临床访谈来进行预问诊和心理疏导。

当前问题ID: {current_question_id}
用户回答: {user_response}
对话历史: {conversation_context}

请以马心宁的身份根据用户的回答进行分析和回应：

1. **情感理解**：识别用户的情感状态
2. **症状评估**：分析是否有心理健康症状指标
3. **下一步决策**：选择最合适的后续问题

可选的问题领域：
- 抑郁症状（情绪低落、兴趣缺失、睡眠、食欲、疲劳等）
- 焦虑症状（过度担心、恐慌、回避行为等）
- 精神病性症状（幻觉、妄想等）
- 物质使用问题
- 创伤经历

请以JSON格式回复，用于debug分析：
{{
    "emotional_analysis": "用户情感状态分析",
    "symptom_indicators": ["发现的症状指标"],
    "risk_assessment": "风险评估",
    "next_question_rationale": "选择下一个问题的理由",
    "user_response": "马心宁的自然共情回应和下一个问题",
    "assessment_complete": false
}}

如果评估应该结束，请设置"assessment_complete": true。
"""
            else:
                # 获取用户主诉
                chief_complaint = state.get("chief_complaint", "")
                
                prompt = f"""
你的名字叫马心宁，是一位温暖、专业的心理咨询师，正在与来访者进行温和的对话了解，进行预问诊和心理疏导。

用户的主诉："{chief_complaint}"
用户刚才说："{user_response}"

请以马心宁的身份像真正的心理咨询师一样回应：
1. 首先表达理解和共情
2. 基于用户的主诉和当前回答，自然地引出下一个相关问题
3. 保持对话的自然流畅，不要显得过于结构化
4. 围绕用户的主要困扰进行深入了解

对话风格要求：
- 温暖、理解、自然
- 像朋友般的专业关怀
- 避免明显的"问卷式"提问
- 根据用户的主诉和回答内容自然延展
- 体现马心宁专业而温暖的个人风格

重点关注领域（但要结合用户主诉进行个性化询问）：
- 与主诉相关的具体症状和表现
- 问题的持续时间和发展过程
- 对日常生活的具体影响
- 情绪状态和心境变化
- 睡眠、食欲、精力等生理方面
- 社交和工作情况
- 既往的应对方式和求助经历

请直接给出自然的对话回应，不需要任何格式标记。如果觉得已经了解足够信息，可以开始总结。

如果需要结束评估，请在回应最后加上"[ASSESSMENT_COMPLETE]"。
"""
            
            print("=" * 50)
            print("🔍 DEBUG - GET_NEXT_QUESTION_RESPONSE LLM CALL")
            print("PROMPT:")
            print(prompt)
            print("=" * 50)
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            print("RESPONSE:")
            print(response.content)
            print("=" * 50)
            
            ai_response = response.content
            
            # 处理debug模式的JSON回复
            if config.DEBUG:
                try:
                    import json
                    debug_data = json.loads(ai_response)
                    
                    # 存储debug信息到状态中
                    if "debug_info" not in state:
                        state["debug_info"] = []
                    
                    state["debug_info"].append({
                        "question_id": current_question_id,
                        "user_input": user_response,
                        "analysis": debug_data
                    })
                    
                    # 检查是否完成评估
                    if debug_data.get("assessment_complete", False):
                        state["assessment_complete"] = True
                        return self._generate_assessment_summary(state)
                    
                    return debug_data.get("user_response", "请继续分享您的感受。")
                    
                except json.JSONDecodeError:
                    # 如果JSON解析失败，直接使用原始回复
                    pass
            
            # 处理正常模式的回复
            if "[ASSESSMENT_COMPLETE]" in ai_response:
                state["assessment_complete"] = True
                ai_response = ai_response.replace("[ASSESSMENT_COMPLETE]", "").strip()
                # 先给出当前回应，然后生成总结
                state["final_response"] = ai_response
                summary_response = self._generate_assessment_summary(state)
                return f"{ai_response}\n\n{summary_response}"
            
            return ai_response
            
        except Exception as e:
            # 如果LLM调用失败，使用简单的规则
            return self._fallback_question_logic(current_question_id, user_response, state)
    
    def _fallback_question_logic(self, current_question_id: str, user_response: str, state: dict) -> str:
        """备用的简单规则逻辑"""
        responses_count = len(state.get("user_responses", {}))
        
        if responses_count >= 3:
            state["assessment_complete"] = True
            return self._generate_simple_summary(state)
        
        # 简单的问题序列
        if current_question_id == "depression_mood":
            return "谢谢您的分享。我想了解一下您平时的生活状态，您最近还像以前一样对自己喜欢的活动感到有趣和开心吗？比如看电影、和朋友聊天、听音乐，或者其他您以前喜欢做的事情？"
        elif "depression" in current_question_id:
            return "了解了。我还想了解一下您最近是否有一些担心或者焦虑的感受？比如对未来的担忧，或者总是放不下心来的事情？"
        else:
            state["assessment_complete"] = True
            return self._generate_simple_summary(state)
    
    def _generate_assessment_summary(self, state: dict) -> str:
        """使用LLM生成评估总结"""
        try:
            responses = state.get("user_responses", {})
            conversation = "\n".join(state.get("conversation_history", []))
            chief_complaint = state.get("chief_complaint", "")
            
            prompt = f"""
基于以下心理健康筛查对话，生成一份专业、温暖的评估总结：

用户主诉：
{chief_complaint}

用户回答记录：
{json.dumps(responses, ensure_ascii=False, indent=2)}

对话历史：
{conversation}

        请生成包含以下内容的总结：

        ## 🔍 评估总结

        ### 用户主诉
        [简要重述用户的主要困扰和求助原因]

        ### 主要发现
        [基于对话总结用户的主要困扰和症状表现，重点分析与主诉相关的内容]

        ### 风险评估
        [评估当前的心理健康风险等级]

        ### 💡 针对性建议
        [基于用户主诉和症状，提供具体的建议和后续步骤]

        ### ⚠️ 重要说明
        - 此评估仅供参考，不能替代专业医疗诊断
        - 如有持续症状，建议咨询专业心理健康服务
        - 如有紧急情况，请立即拨打：400-161-9995

        请保持温暖、支持性的语调，体现对用户主诉的理解和关注。
"""
            
            print("=" * 50)
            print("🔍 DEBUG - GENERATE_ASSESSMENT_SUMMARY LLM CALL")
            print("PROMPT:")
            print(prompt)
            print("=" * 50)
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            print("RESPONSE:")
            print(response.content)
            print("=" * 50)
            
            return response.content
            
        except Exception:
            return self._generate_simple_summary(state)
    
    def _generate_simple_summary(self, state: dict) -> str:
        """生成简单的评估总结"""
        return """## 🔍 评估总结

感谢您完成这次心理健康筛查。

### 💡 建议
- 如有任何困扰或症状持续，建议咨询专业心理健康服务
- 保持良好的作息和生活习惯
- 寻求家人朋友的支持

### ⚠️ 重要说明
- 此评估仅供参考，不能替代专业医疗诊断
- 如有紧急情况，请立即拨打：400-161-9995

**您的心理健康很重要，请不要犹豫寻求专业帮助。**"""

# 全局代理实例
# 可以通过修改这里的workflow_mode参数来切换工作模式：
# - "adaptive": 智能检测模式，根据用户意图自由切换问诊和闲聊
# - "structured": 固定流程模式，先完成问诊再转CBT闲聊
scid5_agent = SCID5Agent(workflow_mode="adaptive")  # 默认使用智能检测模式

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