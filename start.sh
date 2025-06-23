#!/bin/bash

echo "启动SCID-5精神疾病问诊助手..."
echo "请确保已安装依赖: pip install -r requirements.txt"
echo "请确保已配置环境变量，可参考config.env.example"
echo ""
echo "启动Streamlit应用..."

streamlit run src/streamlit_ui.py --server.port 8503
