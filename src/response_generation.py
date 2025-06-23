"""
回应生成模块
包含评估总结生成和各种智能回应功能
"""

import json
from typing import Tuple
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from agent_types import InterviewState
from scid5_knowledge import scid5_kb
from config import config


class ResponseGenerator:
    """回应生成器"""
    
    def __init__(self, llm):
        self.llm = llm
    
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

### 主要发现
- 基于对话内容总结用户的主要困扰和症状表现
- 识别的情感模式和行为特征

### 风险评估
- 当前的心理健康风险等级
- 需要特别关注的方面

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
        workflow_mode = getattr(self, 'workflow_mode', 'adaptive')
        if workflow_mode == "structured":
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
    
    def get_next_question_response(self, current_question_id: str, user_response: str, state: dict) -> str:
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
                        return self.generate_assessment_summary(state)
                    
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
                summary_response = self.generate_assessment_summary(state)
                return f"{ai_response}\n\n{summary_response}"
            
            return ai_response
            
        except Exception as e:
            # 如果LLM调用失败，使用简单的规则
            return self.fallback_question_logic(current_question_id, user_response, state)
    
    def fallback_question_logic(self, current_question_id: str, user_response: str, state: dict) -> str:
        """备用的简单规则逻辑"""
        responses_count = len(state.get("user_responses", {}))
        
        if responses_count >= 3:
            state["assessment_complete"] = True
            return self.generate_simple_summary(state)
        
        # 简单的问题序列
        if current_question_id == "depression_mood":
            return "谢谢您的分享。我想了解一下您平时的生活状态，您最近还像以前一样对自己喜欢的活动感到有趣和开心吗？比如看电影、和朋友聊天、听音乐，或者其他您以前喜欢做的事情？"
        elif "depression" in current_question_id:
            return "了解了。我还想了解一下您最近是否有一些担心或者焦虑的感受？比如对未来的担忧，或者总是放不下心来的事情？"
        else:
            state["assessment_complete"] = True
            return self.generate_simple_summary(state)
    
    def generate_assessment_summary(self, state: dict) -> str:
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
            return self.generate_simple_summary(state)
    
    def generate_simple_summary(self, state: dict) -> str:
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
    
    def fallback_response(self, state: dict, user_message: str) -> Tuple[str, dict]:
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