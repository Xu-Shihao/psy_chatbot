"""
问诊流程模块
包含问诊开始、问题询问和用户回答理解等功能
"""

import json
from datetime import datetime
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from agent import InterviewState
from scid5_knowledge import scid5_kb
from prompts import PromptTemplates


class InterviewFlowHandler:
    """问诊流程处理器"""
    
    def __init__(self, openai_client):
        self.openai_client = openai_client
    
    def _build_conversation_messages(self, state: dict, system_prompt: str, new_user_message: str = None) -> list:
        """
        构建正确格式的对话消息列表: system -> human -> assistant -> human -> assistant...
        
        Args:
            state: 当前状态
            system_prompt: 系统消息内容
            new_user_message: 新的用户消息（可选）
        
        Returns:
            格式正确的消息列表
        """
        messages = [SystemMessage(content=system_prompt)]
        
        # 获取现有的对话消息（跳过系统消息）
        existing_messages = state.get("messages", [])
        
        # 过滤出人类和AI消息，确保交替格式
        conversation_messages = []
        for msg in existing_messages:
            if isinstance(msg, (HumanMessage, AIMessage)):
                conversation_messages.append(msg)
        
        # 添加现有的对话消息
        messages.extend(conversation_messages)
        
        # 如果有新的用户消息，添加到最后
        if new_user_message:
            messages.append(HumanMessage(content=new_user_message))
        
        # 调试输出
        print(f"🔍 DEBUG - InterviewFlow构建的消息序列:")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
            print(f"  [{i}] {msg_type}: {content_preview}...")
        
        return messages
    
    def _build_conversation_context(self, state: dict) -> str:
        """构建对话上下文字符串"""
        conversation_history = state.get("conversation_history", [])
        if not conversation_history:
            return "无对话历史"
        
        # 使用现有的格式化函数获取最近的对话
        from agent import format_conversation_context
        recent_history = format_conversation_context(conversation_history, max_turns=4)
        
        # 转换为字符串格式
        context_lines = []
        for msg in recent_history:
            if msg["role"] == "user":
                context_lines.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                context_lines.append(f"You: {msg['content']}")
        
        return "\n".join(context_lines) if context_lines else "无对话历史"
    
    def start_interview(self, state: InterviewState) -> InterviewState:
        """开始问诊 - 生成自我介绍和引导开场白"""
        system_message = SystemMessage(content=PromptTemplates.COUNSELOR_SYSTEM_PROMPT)
        
        # 生成自我介绍和引导开场白
        try:
            initial_prompt = PromptTemplates.INITIAL_INTERVIEW_PROMPT
            
            print("=" * 50)
            print("🔍 DEBUG - START_INTERVIEW LLM CALL", flush=True)
            print("PROMPT:")
            print(initial_prompt)
            print("=" * 50)
            
            # 使用OpenAI API直接调用
            from agent import call_openai_api
            openai_messages = [{"role": "user", "content": initial_prompt}]
            response_content = call_openai_api(self.openai_client, openai_messages)
            initial_response = response_content.strip()
            
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
        
        # 构建综合分析的系统消息
        next_question_info = self._get_next_question_info(current_question_id, last_user_message, state)
        current_disorder_focus = state.get("current_question_id", "depression_screening")
        
        # 获取最后一条AI回复
        last_ai_response = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                last_ai_response = msg.content
                break
        if not last_ai_response:
            last_ai_response = "开始对话"
        
        # 构建对话上下文
        conversation_context = self._build_conversation_context(state)
        
        # 使用prompts.py中的统一函数
        system_prompt = PromptTemplates.get_comprehensive_analysis_prompt(
            conversation_context=conversation_context,
            last_ai_response=last_ai_response,
            last_user_message=last_user_message,
            current_disorder_focus=current_disorder_focus,
            next_question_info=next_question_info
        )
        
        # 构建正确格式的消息列表
        current_messages = self._build_conversation_messages(
            state, 
            system_prompt, 
            last_user_message
        )
        
        print("=" * 50)
        print("🔍 DEBUG - UNDERSTAND_AND_RESPOND (COMPREHENSIVE) LLM CALL", flush=True)
        print(f"🔍 DEBUG - Messages数量: {len(current_messages)}", flush=True)
        print("=" * 50)
        
        # 转换为OpenAI格式并调用API
        from agent import call_openai_api, messages_to_openai_format
        openai_messages = messages_to_openai_format(current_messages)
        response_content = call_openai_api(self.openai_client, openai_messages)
        
        print("RESPONSE:")
        print(response_content)
        print("=" * 50)
        
        try:
            analysis = json.loads(response_content)
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
        
        # 导入工具函数
        from agent import update_conversation_history_openai
        
        # 更新对话历史 - 使用OpenAI格式和真实的AI回复内容
        last_ai_message = None
        if len(state["messages"]) > 1:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    last_ai_message = msg.content
                    break
        
        # 使用新的工具函数更新对话历史
        updated_history = update_conversation_history_openai(
            state, 
            ai_message=last_ai_message, 
            user_message=last_user_message
        )
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