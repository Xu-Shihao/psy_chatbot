# SCID-5 精神疾病问诊 Chatbot

基于LangGraph的SCID-5结构化临床访谈精神疾病问诊助手。

## 快速开始

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 配置API密钥:
```bash
# 复制配置模板
cp config.env.example .env

# 编辑配置文件，设置您的OpenAI API密钥
# vim .env 或者 nano .env
# 修改 OPENAI_API_KEY=your_openai_api_key_here 为您的真实API密钥
```

3. 启动应用:
```bash
./start.sh
```

或者直接运行:
```bash
PYTHONPATH=./src streamlit run src/streamlit_ui.py
```

## 功能特点

- 🧠 基于SCID-5标准的结构化问诊
- 🤖 LangGraph智能工作流
- 💻 现代化Web界面
- 🚨 自动危机干预检测
- 📊 详细评估报告

## 重要声明

⚠️ 本工具仅供筛查参考，不能替代专业医疗诊断。
如有紧急情况请立即拨打：400-161-9995 