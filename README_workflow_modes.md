# SCID5 Agent 工作流程模式说明

## 概述

SCID5 Agent 支持两种工作流程模式，以适应不同的使用场景和需求：

1. **智能检测模式 (Adaptive Mode)** - 默认模式
2. **固定流程模式 (Structured Mode)** - 强制问诊模式

## 模式详细说明

### 🧠 智能检测模式 (adaptive)

**特点：**
- 系统会自动检测用户的对话意图
- 用户可以自由选择问诊或闲聊
- 支持在问诊和CBT闲聊之间自由切换
- 提供更灵活的用户体验

**工作流程：**
```
用户输入 → 意图检测 → 选择对应模式
    ↓                    ↓
问诊相关内容 → 问诊模式    闲聊内容 → CBT闲聊模式
    ↓                    ↓
评估完成 → 自动转CBT闲聊   继续CBT闲聊
```

**适用场景：**
- 回访用户，可能只想聊天
- 需要灵活对话体验的场景
- 用户明确知道自己想要什么服务
- 个人使用或非正式咨询

### 🏥 固定流程模式 (structured)

**特点：**
- 强制先完成完整的问诊评估
- 无论用户输入什么，都会引导进行问诊
- 问诊完成后自动转入CBT闲聊模式
- 确保完整的评估流程

**工作流程：**
```
用户输入 → 强制问诊模式 → 完整评估流程
    ↓           ↓              ↓
任何内容 → 引导回问诊 → 评估完成 → 自动转CBT闲聊
```

**适用场景：**
- 首次用户，需要完整评估
- 正式的心理健康筛查
- 医疗或专业环境
- 需要标准化流程的场景

## 使用方法

### 方式1：直接创建代理

```python
from src.langgraph_agent import SCID5Agent

# 智能检测模式
agent_adaptive = SCID5Agent(workflow_mode="adaptive")

# 固定流程模式  
agent_structured = SCID5Agent(workflow_mode="structured")
```

### 方式2：使用工厂函数

```python
from src.langgraph_agent import create_agent

# 创建智能检测模式的代理
agent = create_agent("adaptive")

# 创建固定流程模式的代理
agent = create_agent("structured")
```

### 方式3：修改全局实例

```python
# 在 src/langgraph_agent.py 文件末尾修改：
scid5_agent = SCID5Agent(workflow_mode="structured")  # 改为固定流程模式
```

## 对话示例

### 智能检测模式示例

```
用户: "我想聊聊天"
系统: [检测到闲聊意图] → 进入CBT闲聊模式
AI: "我很高兴和您聊天，您想聊什么呢？..."

用户: "我最近很焦虑"  
系统: [检测到问诊需求] → 进入问诊模式
AI: "我理解您的困扰，让我们详细了解一下您的情况..."
```

### 固定流程模式示例

```
用户: "我想聊聊天"
系统: [强制问诊] → 引导进入问诊流程
AI: "我理解您想要交流，让我们先了解一下您的整体情况..."

用户: "我最近很焦虑"
系统: [继续问诊] → 按问诊流程进行
AI: "谢谢您的分享，让我进一步了解您焦虑的具体情况..."

[问诊完成后]
AI: "🌟 问诊评估已完成！现在我们可以进入更轻松的交流环节..."
```

## 配置建议

### 选择智能检测模式的情况：
- ✅ 用户群体多样化，有不同需求
- ✅ 希望提供灵活的用户体验
- ✅ 用户可能只需要情感支持而非评估
- ✅ 个人或非正式使用场景

### 选择固定流程模式的情况：
- ✅ 需要标准化的评估流程
- ✅ 首次用户居多，需要完整了解
- ✅ 医疗或专业咨询环境
- ✅ 需要确保评估质量和完整性

## 动态模式选择

您还可以根据用户类型或场景动态选择模式：

```python
def get_agent_for_scenario(user_type: str, scenario: str) -> SCID5Agent:
    """根据用户类型和场景选择合适的代理模式"""
    
    if user_type == "first_time" or scenario == "clinical":
        # 首次用户或临床场景，使用固定流程
        return create_agent("structured")
    elif user_type == "returning" or scenario == "casual":
        # 回访用户或日常场景，使用智能检测
        return create_agent("adaptive")
    else:
        # 默认使用智能检测
        return create_agent("adaptive")

# 使用示例
agent = get_agent_for_scenario("first_time", "clinical")
```

## 注意事项

1. **模式一致性**: 在同一对话会话中，建议保持使用同一种模式
2. **状态管理**: 不同模式的状态结构相同，可以安全切换
3. **用户体验**: 固定流程模式可能会让某些用户感觉不够灵活
4. **评估质量**: 智能检测模式可能无法保证完整的评估覆盖

## 技术实现

两种模式的主要区别在于 `detect_conversation_mode` 方法的行为：

- **智能检测模式**: 使用LLM分析用户意图，决定进入问诊还是闲聊
- **固定流程模式**: 在评估未完成时，强制进入问诊流程，忽略意图检测结果

这种设计确保了代码的简洁性和可维护性，同时提供了灵活的配置选项。 