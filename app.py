#!/usr/bin/env python3
"""
SCID-5 精神疾病问诊 Chatbot 主入口
"""

import streamlit as st
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入主应用
from streamlit_ui import main

if __name__ == "__main__":
    main() 