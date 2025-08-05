#!/bin/bash

# 创建日志目录
mkdir -p logs

# 获取当前时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
logfile="logs/app_${timestamp}.log"

echo "启动SCID-5精神疾病问诊助手..."
echo "请确保已安装依赖: pip install -r requirements.txt"
echo "请确保已配置环境变量，可参考config.env.example"
echo ""
echo "启动Streamlit应用..."
echo "日志文件保存在: $logfile"

# 将输出重定向到日志文件
streamlit run src/streamlit_ui.py --server.port 8503 2>&1 | tee "$logfile"
