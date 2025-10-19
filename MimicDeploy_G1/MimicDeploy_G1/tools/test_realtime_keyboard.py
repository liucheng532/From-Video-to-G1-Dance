#!/usr/bin/env python3
"""
测试实时键盘控制器
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.realtime_keyboard_controller import RealtimeKeyboardController, CommandType
import time

def main():
    """主函数"""
    print("开始测试实时键盘控制器...")
    print("提示: 按下按键立即生效，松开立即停止")
    print("按 ESC 退出测试\n")
    
    controller = RealtimeKeyboardController()
    
    try:
        while controller.is_running():
            # 更新控制器
            controller.update()
            
            # 获取技能命令
            skill_cmd = controller.get_skill_command()
            if skill_cmd is not None:
                if skill_cmd == CommandType.EXIT:
                    print("收到退出命令，程序结束")
                    break
                else:
                    print(f"✓ 技能命令: {skill_cmd.name}")
            
            # 获取速度命令
            vel_x, vel_y, vel_yaw = controller.get_velocity_command()
            if vel_x != 0 or vel_y != 0 or vel_yaw != 0:
                print(f"  速度: 前后={vel_x:.2f}, 左右={vel_y:.2f}, 旋转={vel_yaw:.2f}", end='\r')
            
            # 短暂睡眠
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n收到中断信号，程序结束")
    finally:
        controller.quit()
        print("测试完成")

if __name__ == "__main__":
    main()

