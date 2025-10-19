# 统一控制器使用说明

## 概述

本项目现在支持手柄和键盘的统一控制。当没有检测到手柄时，系统会自动切换到键盘控制模式，无需修改现有代码。

## 新增文件

### 核心文件
- `common/keyboard_controller.py` - 键盘控制器实现
- `common/unified_controller.py` - 统一控制器，自动检测手柄或键盘
- `deploy_mujoco/deploy_mujoco_unified.py` - 使用统一控制器的示例

### 文档和测试
- `tools/keyboard_mapping.md` - 详细的键盘映射说明
- `tools/test_unified_controller.py` - 控制器测试脚本
- `README_controller.md` - 本说明文档

## 快速开始

### 1. 测试控制器功能
```bash
cd /home/run/code/OpenSource/RoboMimicDeploy_G1
python tools/test_unified_controller.py
```

### 2. 使用新的统一控制器
将现有的deploy文件中的手柄初始化代码：
```python
from common.joystick import JoyStick, JoystickButton
joystick = JoyStick()
```

替换为：
```python
from common.unified_controller import UnifiedController, JoystickButton
controller = UnifiedController(prefer_joystick=True)
```

然后将所有的 `joystick.` 调用替换为 `controller.` 即可。

### 3. 运行示例
```bash
cd deploy_mujoco
python deploy_mujoco_unified.py
```

## 功能特性

### 自动检测
- 优先检测手柄，如果没有手柄则自动使用键盘
- 支持运行时切换控制器（键盘模式下按H键）

### 完全兼容
- 保持与原有 `JoystickButton` 枚举的完全兼容
- 提供相同的API接口：`is_button_pressed()`, `get_axis_value()` 等
- 现有代码无需修改，只需替换初始化部分

### 键盘映射
- 智能的键盘到手柄按键映射
- 支持组合键操作
- 摇杆轴通过WASD和方向键模拟

## 键盘控制映射

### 基础按钮
| 手柄 | 键盘 | 功能 |
|-----|------|------|
| A | 空格 | 确认/跳跃 |
| B | ESC | 取消 |
| X | X | 技能按钮 |
| Y | Y | 技能按钮 |
| L1 | Q | 左肩键 |
| R1 | E | 右肩键 |
| SELECT | Tab | 选择/退出 |
| START | Enter | 开始 |
| HOME | H | 切换控制器 |

### 方向控制
- **左摇杆**: WASD (W=上, A=左, S=下, D=右)
- **右摇杆**: 方向键 (↑=上, ←=左, ↓=下, →=右)

### 技能组合
- **被动模式**: Q + E (L1 + R1)
- **位置重置**: Enter (START)
- **摔倒爬起**: Q + X (L1 + X)
- **行走模式**: E + 空格 (R1 + A)
- **技能1-5**: 各种组合键，详见 `tools/keyboard_mapping.md`

## 使用方法

### 方法1: 直接替换（推荐）
1. 将 `from common.joystick import JoyStick, JoystickButton` 替换为 `from common.unified_controller import UnifiedController, JoystickButton`
2. 将 `joystick = JoyStick()` 替换为 `controller = UnifiedController(prefer_joystick=True)`
3. 将所有 `joystick.` 调用替换为 `controller.`

### 方法2: 使用示例文件
直接使用 `deploy_mujoco_unified.py` 作为模板，复制其控制器使用方式。

## 配置选项

### UnifiedController 参数
```python
controller = UnifiedController(prefer_joystick=True)
```

- `prefer_joystick=True`: 优先使用手柄，没有手柄时使用键盘
- `prefer_joystick=False`: 优先使用键盘，没有键盘时使用手柄

### 控制器切换
在键盘模式下，按 **H键** 可以尝试切换到手柄（如果连接了手柄）。

## 故障排除

### 常见问题

1. **键盘无响应**
   - 确保程序窗口处于焦点状态
   - 检查是否有其他程序占用键盘

2. **无法检测到手柄**
   - 检查手柄连接
   - 确认手柄驱动正常
   - 运行 `python tools/joystick_test.py` 测试手柄

3. **按键映射错误**
   - 检查键盘布局是否为英文
   - 参考 `tools/keyboard_mapping.md` 确认按键映射

### 调试工具

1. **测试统一控制器**:
   ```bash
   python tools/test_unified_controller.py
   ```

2. **测试手柄**:
   ```bash
   python tools/joystick_test.py
   ```

3. **测试键盘控制器**:
   ```bash
   python tools/test_unified_controller.py
   # 选择选项2
   ```

## 自定义配置

### 修改键盘映射
编辑 `common/keyboard_controller.py` 中的 `key_mapping` 字典：

```python
self.key_mapping = {
    K_SPACE: KeyboardButton.A,      # 空格 -> A
    K_ESCAPE: KeyboardButton.B,     # ESC -> B
    # 添加或修改其他映射
}
```

### 修改轴映射
编辑 `axis_mapping` 字典来改变摇杆轴的控制方式。

## 技术细节

### 架构设计
- `KeyboardController`: 实现键盘控制，提供与手柄相同的接口
- `UnifiedController`: 统一控制器，自动检测和切换
- 保持向后兼容性，现有代码无需修改

### 性能考虑
- 键盘控制器使用pygame事件系统，性能良好
- 自动检测只在初始化时进行，运行时开销很小
- 支持实时切换控制器类型

## 贡献

如果需要添加新的控制器类型或修改现有功能，请：

1. 确保新控制器实现相同的接口
2. 更新 `UnifiedController` 以支持新类型
3. 添加相应的测试和文档
4. 保持向后兼容性

