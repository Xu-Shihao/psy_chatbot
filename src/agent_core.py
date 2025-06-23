"""
核心代理类模块
包含 SCID5Agent 类的基础结构和初始化方法
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from agent_types import InterviewState
from config import config


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