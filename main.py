import sys
import os
sys.path.append('src')

from src.streamlit_ui import main

if __name__ == "__main__":
    # 添加debug信息
    print("🚀 启动 SCID-5 问诊系统...")
    print(f"Python路径: {sys.path}")
    print(f"当前工作目录: {os.getcwd()}")
    
    try:
        main()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc() 