"""
配置管理模块
"""
import os
from dotenv import load_dotenv
from typing import Optional

# 加载环境变量
load_dotenv()

class Config:
    """应用配置类"""
    
    # OpenAI API配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    # 模型配置
    MODEL_NAME: str = os.getenv("MODEL_NAME", "qwen/qwen3-8b:free")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    
    # 应用配置
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """验证配置是否完整"""
        if not cls.OPENAI_API_KEY:
            return False
        return True
    
    @classmethod
    def get_openai_config(cls) -> dict:
        """获取OpenAI配置"""
        return {
            "api_key": cls.OPENAI_API_KEY,
            "base_url": cls.OPENAI_API_BASE,
            "model": cls.MODEL_NAME,
            "temperature": cls.TEMPERATURE,
            "max_tokens": cls.MAX_TOKENS
        }

config = Config() 