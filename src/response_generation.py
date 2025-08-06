"""
回应生成模块
包含评估总结生成和各种智能回应功能
"""

import json
from typing import Tuple
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from agent import InterviewState
from scid5_knowledge import scid5_kb
from config import config
from prompts import PromptTemplates


class ResponseGenerator:
    """回应生成器"""
    
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
        print(f"🔍 DEBUG - 构建的消息序列:")
        for i, msg in enumerate(messages):
            msg_type = type(msg).__name__
            content_preview = msg.content[:50] if hasattr(msg, 'content') else str(msg)[:50]
            print(f"  [{i}] {msg_type}: {content_preview}...")
        
        return messages
    
    def _format_conversation_context(self, state: dict) -> str:
        """
        格式化对话历史上下文
        
        Args:
            state: 当前状态
            
        Returns:
            格式化的对话历史字符串
        """
        messages = state.get("messages", [])
        if not messages:
            return "无对话历史"
        
        # 获取最近的几轮对话
        recent_messages = messages[-6:] if len(messages) >= 6 else messages
        
        formatted_lines = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                formatted_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_lines.append(f"You: {msg.content}")
            # 跳过SystemMessage
        
        return "\n".join(formatted_lines) if formatted_lines else "无对话历史"
    
    def generate_summary(self, state: InterviewState) -> InterviewState:
        """生成评估总结"""
        assessment_summary = scid5_kb.generate_assessment_summary()
        current_messages = state.get("messages", []).copy()
        
        # 使用统一的总结prompt
        summary_system_prompt = PromptTemplates.get_assessment_summary_prompt(
            current_messages, state.get("user_responses", {}), assessment_summary
        )
        
        # 构建正确格式的消息列表
        current_messages = self._build_conversation_messages(
            state, 
            summary_system_prompt, 
            "请基于我们的完整对话历史生成最终评估总结。"
        )
        
        print("=" * 50, flush=True)
        print("🔍 DEBUG - GENERATE_SUMMARY LLM CALL", flush=True)
        print(f"🔍 DEBUG - Messages数量: {len(current_messages)}", flush=True)
        print("=" * 50, flush=True)
        
        # 转换为OpenAI格式并调用API
        from agent import call_openai_api, messages_to_openai_format
        openai_messages = messages_to_openai_format(current_messages)
        response_content = call_openai_api(self.openai_client, openai_messages)
        
        print("RESPONSE:", flush=True)
        print(response_content, flush=True)
        print("=" * 50, flush=True)
        
        detailed_summary = response_content
        
        # 根据工作模式添加不同的结束语
        workflow_mode = getattr(self, 'workflow_mode', 'adaptive')
        if workflow_mode == "structured":
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
    
    def get_next_question_response(self, current_question_id: str, user_response: str, state: dict) -> str:
        """根据当前问题和用户回答生成下一个问题"""
        try:
            # 获取当前的messages
            current_messages = state.get("messages", []).copy()
            
            # 根据debug模式调整prompt内容
            if config.DEBUG:
                # 构建对话历史上下文
                conversation_context = self._format_conversation_context(state)
                system_prompt = PromptTemplates.get_debug_response_prompt(
                    current_question_id, user_response, conversation_context
                )
            else:
                # 获取用户主诉和当前症状
                chief_complaint = state.get("chief_complaint", "")
                current_symptoms = state.get("current_symptoms", [])
                
                # 构建对话历史上下文
                conversation_context = self._format_conversation_context(state)
                
                system_prompt = PromptTemplates.get_regular_response_prompt(
                    current_question_id, user_response, conversation_context, 
                    chief_complaint, current_symptoms
                )
            
            # 构建正确格式的消息列表
            current_messages = self._build_conversation_messages(
                state, 
                system_prompt, 
                user_response
            )
            
            print("=" * 50, flush=True)
            print("🔍 DEBUG - GET_NEXT_QUESTION_RESPONSE LLM CALL", flush=True)
            print(f"🔍 DEBUG - Messages数量: {len(current_messages)}", flush=True)
            print("=" * 50, flush=True)
            
            # 转换为OpenAI格式并调用API
            from agent import call_openai_api, messages_to_openai_format
            openai_messages = messages_to_openai_format(current_messages)
            response_content = call_openai_api(self.openai_client, openai_messages)
            
            print("RESPONSE:", flush=True)
            print(response_content, flush=True)
            print("=" * 50, flush=True)
            
            ai_response = response_content
            
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
            chief_complaint = state.get("chief_complaint", "")
            current_messages = state.get("messages", []).copy()
            
            system_prompt = f"""
基于以下心理健康筛查对话，生成一份专业、温暖的评估总结：

用户主诉：
{chief_complaint}

用户回答记录：
{json.dumps(responses, ensure_ascii=False, indent=2)}

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
            
            # 构建正确格式的消息列表
            current_messages = self._build_conversation_messages(
                state, 
                system_prompt, 
                "请基于我们的对话历史生成评估总结。"
            )
            
            print("=" * 50, flush=True)
            print("🔍 DEBUG - GENERATE_ASSESSMENT_SUMMARY LLM CALL", flush=True)
            print(f"🔍 DEBUG - Messages数量: {len(current_messages)}", flush=True)
            print("=" * 50, flush=True)
            
            # 转换为OpenAI格式并调用API
            from agent import call_openai_api, messages_to_openai_format
            openai_messages = messages_to_openai_format(current_messages)
            response_content = call_openai_api(self.openai_client, openai_messages)
            
            print("RESPONSE:", flush=True)
            print(response_content, flush=True)
            print("=" * 50, flush=True)
            
            return response_content
            
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
            # 构建正确格式的消息列表
            current_messages = self._build_conversation_messages(
                state, 
                PromptTemplates.CBT_THERAPIST_SYSTEM_PROMPT, 
                user_message
            )
            
            print("=" * 50, flush=True)
            print("🔍 DEBUG - FALLBACK_RESPONSE LLM CALL", flush=True)
            print(f"🔍 DEBUG - Messages数量: {len(current_messages)}", flush=True)
            print("=" * 50, flush=True)
            
            # 转换为OpenAI格式并调用API
            from agent import call_openai_api, messages_to_openai_format
            openai_messages = messages_to_openai_format(current_messages)
            response_content = call_openai_api(self.openai_client, openai_messages)
            
            print("RESPONSE:", flush=True)
            print(response_content, flush=True)
            print("=" * 50, flush=True)
            
            # 创建 AI 回复消息
            ai_response_message = AIMessage(content=response_content)
            
            # 更新状态
            updated_state = state.copy()
            # 导入工具函数并使用OpenAI格式更新conversation_history
            from agent import update_conversation_history_openai
            updated_history = update_conversation_history_openai(state, user_message=user_message)
            updated_state["conversation_history"] = updated_history
            
            # 更新messages：从current_messages中移除最后添加的用户消息，然后添加到原始messages中
            original_messages = state.get("messages", []).copy()
            original_messages.append(HumanMessage(content=user_message))
            original_messages.append(ai_response_message)
            updated_state["messages"] = original_messages
            updated_state["final_response"] = response_content
            
            print(f"🔍 DEBUG - 添加AI回复到messages，新的messages数量: {len(updated_state['messages'])}", flush=True)
            
            return response_content, updated_state
            
        except Exception as e:
            fallback_msg = "抱歉，我现在无法很好地回应您。请您再试一次，或者告诉我您想要进行问诊还是想要闲聊。"
            return fallback_msg, state