# 测试文件夹

这个文件夹包含项目的各种测试文件。

## 测试文件说明

- `test_conversation_modes.py`: 测试对话模式识别功能，包括：
  - 问诊模式和闲聊模式的识别
  - CBT疗愈师功能
  - 状态管理和模式切换

## 运行测试

### 从项目根目录运行：

```bash
# 运行对话模式测试
python tests/test_conversation_modes.py
```

### 从tests目录运行：

```bash
cd tests
python test_conversation_modes.py
```

## 测试覆盖范围

- ✅ 对话模式检测
- ✅ CBT疗愈师激活
- ✅ 问诊流程
- ✅ 状态管理
- ✅ 错误处理

## 添加新测试

在添加新测试文件时，请：

1. 以 `test_` 开头命名文件
2. 包含详细的测试描述
3. 添加必要的错误处理
4. 更新此README文件 