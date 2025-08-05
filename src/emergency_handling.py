"""
紧急情况处理模块
包含风险评估和紧急情况响应功能
"""

import json
from langchain.schema import HumanMessage, AIMessage

from agent import InterviewState
from prompts import PromptTemplates


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
            emergency_keywords = ["自杀", "结束生命", "不想活", "死了算了", "自杀计划"]
            emergency_detected = any(keyword in last_response for keyword in emergency_keywords)
        
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