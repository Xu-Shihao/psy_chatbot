"""
紧急情况处理模块
包含风险评估和紧急情况响应功能
"""

import json
from langchain.schema import HumanMessage, AIMessage

from agent_types import InterviewState


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