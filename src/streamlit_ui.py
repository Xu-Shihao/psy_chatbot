"""
基于Streamlit的SCID-5问诊界面
提供现代化的Web UI进行精神疾病问诊
"""

import streamlit as st
from datetime import datetime
from typing import Optional
import sys
import os
import asyncio
        
# 添加当前目录到Python路径以支持导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from config import config
from langgraph_agent import scid5_agent, InterviewState
from factory import create_agent

# 页面配置
st.set_page_config(
    page_title="灵溪智伴：SCID-5 精神疾病问诊助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        max-width: 80%;
        background: #fff3e0;
        color: #000000;
    }
    
    .user-message {
        background: #fff3e0;
        margin-left: auto;
        border-left: 4px solid #667eea;
        color: #000000;
    }
    
    .ai-message {
        background: #fff3e0;
        margin-right: auto;
        border-left: 4px solid #764ba2;
        color: #000000;
    }
    
    .emergency-alert {
        background: #ffebee;
        border: 2px solid #f44336;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .progress-indicator {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .disclaimer {
        background: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #000000;
    }
    
    .disclaimer h4 {
        color: #000000;
    }
    
    .disclaimer ul {
        color: #000000;
    }
    
    .disclaimer li {
        color: #000000;
    }
    
    .cbt-mode-indicator {
        background: linear-gradient(90deg, #e8f5e8 0%, #f0f8ff 100%);
        border: 1px solid #4caf50;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        color: #2e7d32;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "interview_state" not in st.session_state:
        st.session_state.interview_state = None
    
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    
    if "assessment_complete" not in st.session_state:
        st.session_state.assessment_complete = False
    
    if "cbt_chat_started" not in st.session_state:
        st.session_state.cbt_chat_started = False
    
    if "workflow_mode" not in st.session_state:
        st.session_state.workflow_mode = "adaptive"
    
    if "agent_instance" not in st.session_state:
        st.session_state.agent_instance = None

def display_header():
    """显示页面头部"""
    st.markdown("""
    <div class="main-header">
        <h1> 🧚🏻‍♀️ 灵溪智伴：SCID-5 精神疾病问诊助手</h1>
        <p>基于结构化临床访谈的专业心理健康评估工具</p>
    </div>
    """, unsafe_allow_html=True)

def display_disclaimer():
    """显示免责声明"""
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ 重要声明</h4>
        <ul>
            <li>本工具可进行心理健康筛查和CBT疗愈对话，支持智能检测和固定流程两种工作模式</li>
            <li>本工具仅目的用于心理健康筛查，不能替代专业医疗诊断</li>
            <li>如有严重症状或紧急情况，请立即寻求专业医疗帮助</li>
            <li>评估结果仅供参考，最终诊断需由专业医疗人员确定</li>
            <li>如有自杀或自伤想法，请立即联系紧急求助热线</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_progress_indicator(current_phase: str):
    """显示进度指示器"""
    phases = {
        "开始": 1,
        "抑郁筛查": 2,
        "焦虑筛查": 3,
        "精神病性症状": 4,
        "物质使用": 5,
        "评估完成": 6
    }
    
    current_step = phases.get(current_phase, 1)
    progress = (current_step - 1) / (len(phases) - 1)
    
    st.markdown(f"""
    <div class="progress-indicator">
        <h4>评估进度</h4>
        <p>当前阶段: {current_phase}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(progress)

def display_chat_message(message: str, is_user: bool = False):
    """显示聊天消息"""
    css_class = "user-message" if is_user else "ai-message"
    icon = "👤" if is_user else "🤖"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{icon} {'您' if is_user else 'AI助手'}:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_emergency_alert():
    """显示紧急情况警报"""
    st.markdown("""
    <div class="emergency-alert">
        <h3>🚨 紧急情况检测</h3>
        <p><strong>如果您正在考虑自伤或自杀，请立即寻求帮助：</strong></p>
        <ul>
            <li>📞 全国心理援助热线：400-161-9995</li>
            <li>📞 北京危机干预热线：400-161-9995</li>
            <li>📞 上海心理援助热线：021-34289888</li>
            <li>🏥 立即前往最近的急诊科</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def process_user_input(user_input: str):
    """处理用户输入 - 使用LangGraph Agent"""
    try:
        # DEBUG: 打印用户输入
        print(f"🔍 DEBUG: 用户输入 = {user_input}", flush=True)
        
        # 更新会话状态
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 根据选择的模式创建或获取agent实例
        if st.session_state.agent_instance is None:
            print(f"🔍 DEBUG: 创建新的agent实例，模式: {st.session_state.workflow_mode}", flush=True)
            st.session_state.agent_instance = create_agent(st.session_state.workflow_mode)
        
        current_agent = st.session_state.agent_instance
        
        # 使用选定的 Agent 处理用户输入（同步版本）
        try:
            ai_response, updated_state = current_agent.process_message_sync(
                user_input, st.session_state.interview_state
            )
            
            # 更新状态
            st.session_state.interview_state = updated_state
            
            # 检查是否有final_response字段（CBT疗愈师响应）
            if updated_state.get("final_response"):
                ai_response = updated_state["final_response"]
                print(f"🔍 DEBUG: 使用CBT疗愈师响应 = {ai_response}", flush=True)
            
            # 检查是否评估完成
            if updated_state.get("assessment_complete", False):
                st.session_state.assessment_complete = True
            
            # 检查紧急情况
            if updated_state.get("emergency_situation", False):
                display_emergency_alert()
            
            print(f"🔍 DEBUG: AI回复 = {ai_response}", flush=True)
            
        except Exception as llm_error:
            print(f"❌ DEBUG: LLM调用失败: {llm_error}", flush=True)
            # 如果LLM调用失败，使用备用回复
            ai_response = f"抱歉，AI服务暂时不可用。错误信息：{str(llm_error)}"
            
            # 基本状态管理
            if not st.session_state.interview_state:
                st.session_state.interview_state = {
                    "messages": [],
                    "current_question_id": None,
                    "user_responses": {},
                    "assessment_complete": False,
                    "emergency_situation": False,
                    "summary": "",
                    "conversation_mode": "idle",
                    "chat_therapist_active": False,
                    "current_topic": "初始化",
                    "is_follow_up": False
                }
        
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        return ai_response
        
    except Exception as e:
        error_msg = f"处理消息时出错: {str(e)}"
        print(f"❌ DEBUG: {error_msg}", flush=True)
        st.error(error_msg)
        import traceback
        traceback_str = traceback.format_exc()
        print(f"❌ DEBUG: 详细错误:\n{traceback_str}", flush=True)
        st.error(f"详细错误: {traceback_str}")
        return "抱歉，处理您的消息时出现了错误。请重试。"

def display_sidebar():
    """显示侧边栏"""
    with st.sidebar:
        st.header("📋 评估信息")
        
        # 工作流程模式选择
        st.markdown("### 🎯 工作流程模式")
        workflow_options = {
            "adaptive": "🤖 智能检测模式",
            "structured": "📋 固定流程模式"
        }
        
        selected_mode = st.selectbox(
            "选择工作流程模式:",
            options=list(workflow_options.keys()),
            format_func=lambda x: workflow_options[x],
            index=0 if st.session_state.workflow_mode == "adaptive" else 1,
            key="workflow_selector",
            help="选择对话策略模式"
        )
        
        # 显示模式说明
        if selected_mode == "adaptive":
            st.info("🤖 **智能检测模式**: 根据用户意图自由切换问诊和闲聊")
        else:
            st.info("📋 **固定流程模式**: 先完成问诊再转CBT闲聊")
        
        # 检查是否需要重新创建agent
        if selected_mode != st.session_state.workflow_mode:
            st.session_state.workflow_mode = selected_mode
            st.session_state.agent_instance = None  # 重置agent实例
            # 如果已经开始对话，提示用户重新开始
            if st.session_state.interview_started:
                st.warning("⚠️ 模式已更改，建议重新开始评估以应用新模式")
        
        st.markdown("---")
        
        # 配置状态
        try:
            if config.validate():
                st.success("✅ 系统配置正常")
            else:
                st.error("❌ 请检查API配置")
        except Exception as e:
            st.error(f"❌ 配置检查失败: {e}")
        
        # Debug模式控制
        st.markdown("---")
        current_debug = st.checkbox(
            "🔍 Debug模式", 
            value=config.DEBUG, 
            help="开启后显示AI的决策分析过程"
        )
        
        # 更新config（这只影响UI显示，不修改环境变量）
        if current_debug != config.DEBUG:
            config.DEBUG = current_debug
            if current_debug:
                st.info("✅ Debug模式已开启 - 将显示AI决策分析")
            else:
                st.info("ℹ️ Debug模式已关闭 - 将使用自然对话模式")
        
        # 当前症状类型显示
        if st.session_state.interview_state:
            current_topic = st.session_state.interview_state.get("current_topic", "未知")
            current_question_id = st.session_state.interview_state.get("current_question_id", "")
            
            # 映射问题ID到症状类型的中文名称
            topic_mapping = {
                "depression_screening": "🔵 抑郁症筛查",
                "anxiety_screening": "🟠 焦虑症筛查",
                "ocd_screening": "🟣 强迫症筛查",
                "ptsd_screening": "🟫 创伤后应激障碍筛查",
                "psychotic_screening": "⚪ 精神病性障碍筛查",
                "initial": "🔄 初始问诊",
                "initial_screening": "🔄 初始筛查"
            }
            
            # 显示当前症状类型
            display_topic = topic_mapping.get(current_question_id, current_topic)
            if current_question_id or current_topic != "未知":
                st.markdown("### 🎯 当前评估症状")
                st.info(f"**{display_topic}**")
                
                # 显示问题ID（在debug模式下）
                if config.DEBUG and current_question_id:
                    st.caption(f"问题ID: {current_question_id}")
        
        # DEBUG面板
        with st.expander("🔧 Debug信息", expanded=config.DEBUG):
            if st.session_state.interview_state:
                # 显示AI决策分析
                if "debug_info" in st.session_state.interview_state:
                    st.markdown("#### 🧠 AI决策分析")
                    for i, debug_item in enumerate(st.session_state.interview_state["debug_info"]):
                        st.markdown(f"**📍 对话轮次 {i+1} - {debug_item['question_id']}**")
                        st.markdown(f"**用户输入:** {debug_item['user_input']}")
                        
                        analysis = debug_item.get('analysis', {})
                        if analysis:
                            st.markdown("**AI分析:**")
                            st.markdown(f"- **情感分析:** {analysis.get('emotional_analysis', 'N/A')}")
                            st.markdown(f"- **症状指标:** {', '.join(analysis.get('symptom_indicators', []))}")
                            st.markdown(f"- **风险评估:** {analysis.get('risk_assessment', 'N/A')}")
                            st.markdown(f"- **问题选择理由:** {analysis.get('next_question_rationale', 'N/A')}")
                        
                        st.markdown("---")  # 分隔线
                
                st.markdown("#### 📊 完整状态")
                # 过滤敏感信息，只显示重要的状态
                filtered_state = {
                    "current_question_id": st.session_state.interview_state.get("current_question_id"),
                    "responses_count": len(st.session_state.interview_state.get("user_responses", {})),
                    "assessment_complete": st.session_state.interview_state.get("assessment_complete", False),
                    "emergency_situation": st.session_state.interview_state.get("emergency_situation", False),
                    "conversation_mode": st.session_state.interview_state.get("conversation_mode", "idle"),
                    "chat_therapist_active": st.session_state.interview_state.get("chat_therapist_active", False),
                    "current_topic": st.session_state.interview_state.get("current_topic", "未知"),
                    "is_follow_up": st.session_state.interview_state.get("is_follow_up", False)
                }
                st.json(filtered_state)
                
                # 显示对话模式状态
                st.markdown("#### 🎭 对话模式状态")
                mode = st.session_state.interview_state.get("conversation_mode", "idle")
                cbt_active = st.session_state.interview_state.get("chat_therapist_active", False)
                current_topic = st.session_state.interview_state.get("current_topic", "未知")
                is_follow_up = st.session_state.interview_state.get("is_follow_up", False)
                
                mode_display = {
                    "idle": "🤔 待机中",
                    "interview": "🔍 问诊模式",
                    "chat": "💬 闲聊模式",
                    "continue_interview": "🔄 继续问诊",
                    "assessment_complete": "✅ 评估完成，闲聊模式"
                }
                
                st.write(f"**当前模式**: {mode_display.get(mode, mode)}")
                st.write(f"**当前话题**: {current_topic}")
                
                if is_follow_up:
                    st.info("🔄 当前处于追问状态")
                else:
                    st.write("📝 正常问答状态")
                
                if cbt_active:
                    st.success("🌟 CBT疗愈师已激活")
                
                # 显示当前工作流程模式
                st.markdown("#### ⚙️ 当前工作流程模式")
                current_mode = st.session_state.get("workflow_mode", "adaptive")
                mode_display = {
                    "adaptive": "🤖 智能检测模式",
                    "structured": "📋 固定流程模式"
                }
                st.write(f"**当前模式**: {mode_display.get(current_mode, current_mode)}")
                
                # 显示模式检测结果
                detection_result = st.session_state.interview_state.get("mode_detection_result", {})
                if detection_result:
                    st.markdown("#### 🔍 模式检测分析")
                    st.write(f"**检测结果**: {detection_result.get('detected_mode', 'N/A')}")
                    st.write(f"**置信度**: {detection_result.get('confidence', 0):.2f}")
                    st.write(f"**理由**: {detection_result.get('reason', 'N/A')}")
                    if detection_result.get('key_indicators'):
                        st.write(f"**关键指标**: {', '.join(detection_result['key_indicators'])}")
            else:
                st.info("暂无状态信息")
            
            st.markdown("#### 🎛️ Session State")
            debug_info = {
                "interview_started": st.session_state.get("interview_started", False),
                "assessment_complete": st.session_state.get("assessment_complete", False),
                "cbt_chat_started": st.session_state.get("cbt_chat_started", False),
                "messages_count": len(st.session_state.get("messages", [])),
                "debug_mode": config.DEBUG
            }
            st.json(debug_info)
        
        st.markdown("---")
        
        # 评估统计和报告显示
        if st.session_state.interview_state:
            responses_count = len(st.session_state.interview_state.get("user_responses", {}))
            st.metric("已回答问题", responses_count)
            
            # 检查是否完成评估
            if st.session_state.interview_state.get("assessment_complete", False):
                st.session_state.assessment_complete = True
                
                # 显示评估总结报告
                st.markdown("#### 📋 评估总结报告")
                summary = st.session_state.interview_state.get("summary", "")
                if summary:
                    st.markdown(summary)
                else:
                    st.info("正在生成评估总结...")
                
                # 显示主要发现
                chief_complaint = st.session_state.interview_state.get("chief_complaint", "")
                if chief_complaint:
                    st.markdown("**用户主诉:**")
                    st.write(chief_complaint)
                
                # 显示风险评估
                risk_level = st.session_state.interview_state.get("risk_level", "low")
                risk_indicators = st.session_state.interview_state.get("risk_indicators", [])
                
                risk_colors = {
                    "low": "🟢",
                    "medium": "🟡", 
                    "high": "🔴"
                }
                st.markdown(f"**风险等级:** {risk_colors.get(risk_level, '🟢')} {risk_level.upper()}")
                
                if risk_indicators:
                    st.markdown("**关注点:**")
                    for indicator in risk_indicators:
                        st.write(f"• {indicator}")
                
                st.markdown("---")
        
        st.markdown("---")
        
        # 操作按钮
        if st.button("🔄 重新开始评估"):
            # 保存当前选择的workflow_mode
            current_mode = st.session_state.get("workflow_mode", "adaptive")
            st.session_state.clear()
            st.session_state.workflow_mode = current_mode
            initialize_session_state()
            st.rerun()
        
        if st.button("📊 下载评估报告"):
            if st.session_state.interview_state and st.session_state.assessment_complete:
                report = generate_assessment_report()
                st.download_button(
                    label="下载报告",
                    data=report,
                    file_name=f"assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        st.markdown("---")
        
        # 帮助信息
        with st.expander("📚 使用说明"):
            st.markdown("""
            **如何使用:**
            1. 点击"开始评估"按钮
            2. 诚实回答AI助手的问题
            3. 根据提示进行对话
            4. 完成后查看评估总结
            
            **注意事项:**
            - 请如实回答所有问题
            - 如有紧急情况立即寻求专业帮助
            - 评估结果仅供参考
            """)
        
        with st.expander("🆘 紧急求助"):
            st.markdown("""
            **紧急求助热线:**
            - 全国：400-161-9995
            - 北京：400-161-9995  
            - 上海：021-34289888
            
            **如有紧急情况，请立即拨打以上热线或前往医院急诊科**
            """)

def generate_assessment_report() -> str:
    """生成评估报告"""
    if not st.session_state.interview_state:
        return "暂无评估数据"
    
    state = st.session_state.interview_state
    report = f"""
SCID-5 精神疾病评估报告
============================

评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

用户回答记录:
{'-' * 40}
"""
    
    for q_id, response in state.get("user_responses", {}).items():
        report += f"问题 {q_id}: {response}\n"
    
    report += f"""
{'-' * 40}

评估总结:
{state.get("summary", "暂无总结")}

重要声明:
- 此评估仅供参考，不能替代专业医疗诊断
- 如有症状持续或加重，请及时就医
- 紧急情况请立即寻求专业帮助
"""
    
    return report

def main():
    """主函数"""
    initialize_session_state()
    
    # 显示头部
    display_header()
    
    # 显示免责声明
    display_disclaimer()
    
    # 显示侧边栏
    display_sidebar()
    
    # 主要对话界面
    st.markdown("## 💭 对话界面")
    
    # 显示历史消息
    for message in st.session_state.messages:
        display_chat_message(
            message["content"], 
            is_user=(message["role"] == "user")
        )
    
    # 开始评估按钮
    if not st.session_state.interview_started:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 开始心理健康评估", use_container_width=True):
                st.session_state.interview_started = True
                # 使用LangGraph开始评估
                with st.spinner("AI正在初始化评估..."):
                    process_user_input("开始评估")
                st.rerun()
    
    # 用户输入界面
    elif not st.session_state.assessment_complete:
        user_input = st.chat_input("请输入您的回答...")
        
        if user_input:
            # 处理用户输入
            with st.spinner("AI正在分析您的回答..."):
                process_user_input(user_input)
            st.rerun()
    
    # 评估完成后的CBT疗愈模式
    else:
        # 如果还没有开始CBT对话，显示完整的评估完成界面
        if not st.session_state.cbt_chat_started:
            st.success("✅ 评估已完成！现在进入CBT疗愈模式，我会陪伴您进行心理疏导。")
            
            # 提供下载报告的选项
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 下载完整报告"):
                    report = generate_assessment_report()
                    st.download_button(
                        label="点击下载",
                        data=report,
                        file_name=f"assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                if st.button("🔄 开始新的评估"):
                    # 保存当前选择的workflow_mode
                    current_mode = st.session_state.get("workflow_mode", "adaptive")
                    st.session_state.clear()
                    st.session_state.workflow_mode = current_mode
                    initialize_session_state()
                    st.rerun()
            
            st.markdown("---")
            st.markdown("💬 **继续对话** - 我会为您提供心理支持和疏导")
        else:
            # 如果已经开始CBT对话，显示简洁的状态和快捷操作
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("""
                <div class="cbt-mode-indicator">
                    💬 CBT疗愈对话进行中...
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # 紧凑的下载报告按钮
                if st.button("📄 报告", key="download_report_compact", help="下载评估报告"):
                    report = generate_assessment_report()
                    st.download_button(
                        label="📥 下载",
                        data=report,
                        file_name=f"assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key="download_report_compact_btn"
                    )
            with col3:
                # 重新开始按钮
                if st.button("🔄 重新开始", key="restart_compact", help="开始新的评估"):
                    # 保存当前选择的workflow_mode
                    current_mode = st.session_state.get("workflow_mode", "adaptive")
                    st.session_state.clear()
                    st.session_state.workflow_mode = current_mode
                    initialize_session_state()
                    st.rerun()
        
        # CBT疗愈对话输入框
        user_input = st.chat_input("请告诉我您想聊的话题，或者询问关于心理健康的问题...")
        
        if user_input:
            # 标记CBT对话已开始
            if not st.session_state.cbt_chat_started:
                st.session_state.cbt_chat_started = True
            
            # 处理用户输入，但现在是CBT疗愈模式
            with st.spinner("CBT疗愈师正在回应..."):
                process_user_input(user_input)
            st.rerun()

if __name__ == "__main__":
    main() 