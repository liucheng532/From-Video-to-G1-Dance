#!/usr/bin/env python3
"""
统一控制器测试脚本
测试手柄和键盘控制器的自动切换功能
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.unified_controller import UnifiedController, JoystickButton
import time

def test_controller():
    """测试统一控制器功能"""
    print("=== 统一控制器测试 ===")
    print("正在初始化控制器...")
    
    try:
        # 创建统一控制器
        controller = UnifiedController(prefer_joystick=True)
        print(f"✓ 控制器初始化成功: {controller.get_controller_name()}")
        
        # 显示控制说明
        if controller.get_controller_type() == 0:  # 手柄
            print("\n手柄控制模式:")
            print("  使用手柄按钮和摇杆进行控制")
            print("  按SELECT按钮退出测试")
        else:  # 键盘
            print("\n键盘控制模式:")
            print("  空格=A, ESC=B, X=X, Y=Y")
            print("  Q=L1, E=R1, Tab=SELECT, Enter=START")
            print("  WASD=左摇杆, 方向键=右摇杆")
            print("  按Tab键退出测试")
            print("  按H键可以尝试切换到手柄")
        
        print("\n开始监控输入... (按对应退出键退出)")
        
        # 监控输入
        while True:
            controller.update()
            
            # 检查退出条件
            if controller.is_button_pressed(JoystickButton.SELECT):
                print("\n检测到退出信号，正在退出...")
                break
            
            # 检查控制器切换
            if controller.get_controller_type() == 1 and controller.is_button_released(JoystickButton.HOME):
                print("\n尝试切换控制器...")
                if controller.switch_controller():
                    print(f"控制器已切换: {controller.get_controller_name()}")
                else:
                    print("控制器切换失败")
            
            # 显示按钮状态
            pressed_buttons = []
            for button in JoystickButton:
                if controller.is_button_pressed(button.value):
                    pressed_buttons.append(button.name)
            
            if pressed_buttons:
                print(f"按下按钮: {', '.join(pressed_buttons)}")
            
            # 显示轴状态
            axis_values = []
            for i in range(4):  # 4个轴
                value = controller.get_axis_value(i)
                if abs(value) > 0.1:  # 只显示有意义的轴值
                    axis_values.append(f"轴{i}: {value:.2f}")
            
            if axis_values:
                print(f"摇杆: {', '.join(axis_values)}")
            
            # 显示十字键状态
            hat = controller.get_hat_direction(0)
            if hat != (0, 0):
                print(f"十字键: {hat}")
            
            time.sleep(0.1)
    
    except RuntimeError as e:
        print(f"✗ 控制器初始化失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n用户中断测试")
    finally:
        try:
            controller.quit()
            print("✓ 控制器资源已清理")
        except:
            pass
    
    return True

def test_keyboard_only():
    """测试仅键盘模式"""
    print("\n=== 键盘控制器单独测试 ===")
    
    try:
        from common.keyboard_controller import KeyboardController, KeyboardButton
        
        controller = KeyboardController()
        print("✓ 键盘控制器初始化成功")
        
        print("键盘控制说明:")
        print("  空格=A, ESC=B, X=X, Y=Y")
        print("  Q=L1, E=R1, Tab=SELECT, Enter=START")
        print("  WASD=左摇杆, 方向键=右摇杆")
        print("  按Tab键退出测试")
        
        print("\n开始监控键盘输入...")
        
        while True:
            controller.update()
            
            if controller.is_button_pressed(KeyboardButton.SELECT):
                print("\n检测到退出信号，正在退出...")
                break
            
            # 显示按钮状态
            pressed_buttons = []
            for button in KeyboardButton:
                if controller.is_button_pressed(button.value):
                    pressed_buttons.append(button.name)
            
            if pressed_buttons:
                print(f"按下按钮: {', '.join(pressed_buttons)}")
            
            # 显示轴状态
            axis_values = []
            for i in range(4):
                value = controller.get_axis_value(i)
                if abs(value) > 0.1:
                    axis_values.append(f"轴{i}: {value:.2f}")
            
            if axis_values:
                print(f"摇杆: {', '.join(axis_values)}")
            
            time.sleep(0.1)
    
    except Exception as e:
        print(f"✗ 键盘控制器测试失败: {e}")
        return False
    finally:
        try:
            controller.quit()
        except:
            pass
    
    return True

if __name__ == "__main__":
    print("选择测试模式:")
    print("1. 统一控制器测试 (自动检测手柄/键盘)")
    print("2. 仅键盘控制器测试")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            test_controller()
        elif choice == "2":
            test_keyboard_only()
        else:
            print("无效选择，运行统一控制器测试")
            test_controller()
    
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")

