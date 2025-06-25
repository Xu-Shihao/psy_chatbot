"""
工作流构建模块
包含LangGraph工作流的创建和流程控制逻辑
"""

from langgraph.graph import StateGraph, END

from agent_types import InterviewState
from scid5_knowledge import scid5_kb


class WorkflowBuilder:
    """工作流构建器"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def create_workflow(self) -> StateGraph:
        """创建问诊工作流"""
        workflow = StateGraph(InterviewState)
        
        # 添加节点
        workflow.add_node("start_interview", self.agent.start_interview)
        workflow.add_node("detect_conversation_mode", self.agent.detect_conversation_mode)
        workflow.add_node("chat_therapist_response", self.agent.chat_therapist_response)
        workflow.add_node("understand_and_respond", self.agent.understand_and_respond)
        workflow.add_node("ask_question", self.agent.ask_question)
        workflow.add_node("check_emergency", self.agent.check_emergency)
        workflow.add_node("generate_summary", self.agent.generate_summary)
        workflow.add_node("emergency_response", self.agent.emergency_response)
        
        # 设置入口点
        workflow.set_entry_point("start_interview")
        
        # 定义边
        workflow.add_edge("start_interview", "detect_conversation_mode")
        workflow.add_conditional_edges(
            "detect_conversation_mode",
            self.should_continue_after_mode_detection,
            {
                "interview": "understand_and_respond",  # 直接进入理解和回应阶段
                "chat": "chat_therapist_response",
                "assessment_complete": "chat_therapist_response"
            }
        )
        workflow.add_edge("chat_therapist_response", END)
        workflow.add_conditional_edges(
            "understand_and_respond",
            self.should_continue_after_understand_and_respond,
            {
                "continue": "check_emergency",
                "emergency": "emergency_response",
                "complete": "generate_summary",
                "wait_response": END
            }
        )
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
    
    def should_continue_after_mode_detection(self, state: InterviewState) -> str:
        """决定模式检测后的下一步"""
        # 如果是结构化模式且未完成评估，强制进入问诊流程
        if (self.agent.workflow_mode == "structured" and 
            not state.get("assessment_complete", False)):
            print(f"🔒 工作流强制路由到问诊模式 - 结构化模式未完成评估")
            return "interview"
        
        # 智能检测模式的原有逻辑
        mode = state.get("conversation_mode", "interview")
        
        if mode == "chat":
            return "chat"
        elif mode in ["interview", "continue_interview"]:
            return "interview"  # 统一映射到interview流程
        elif mode == "assessment_complete":
            return "assessment_complete"  
        else:
            return "interview"  # 默认进入问诊模式
    
    def should_continue_after_understand_and_respond(self, state: InterviewState) -> str:
        """理解和回应后的流程控制"""
        if state["emergency_situation"]:
            return "emergency"
        elif state["assessment_complete"]:
            return "complete"
        else:
            return "continue"
    
    def should_continue_after_question(self, state: InterviewState) -> str:
        """问题后的流程控制（保留用于兼容性）"""
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
                if hasattr(msg, 'content'):
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