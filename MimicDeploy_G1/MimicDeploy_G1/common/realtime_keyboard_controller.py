"""
实时键盘控制器 - 无需回车确认，按下即响应，松开即停止
使用pynput库实现实时按键监听
"""
from pynput import keyboard
from enum import IntEnum
import threading
import time

class CommandType(IntEnum):
    """命令类型枚举"""
    PASSIVE = 0        # 被动模式
    POS_RESET = 1      # 位置重置
    STAND_UP = 2       # 站起来
    LOCO = 3           # 运动模式
    SKILL_1 = 4        # 技能1 (舞蹈)
    SKILL_2 = 5        # 技能2 (功夫2)
    # SKILL_3, SKILL_4, SKILL_5 已删除
    MOTION_TRACKING = 9  # 动作跟踪
    EXIT = 99          # 退出

class RealtimeKeyboardController:
    """实时键盘控制器 - 按下即响应，松开即停"""
    
    def __init__(self):
        """初始化实时键盘控制器"""
        self.running = True
        
        # 速度控制参数
        self.vel_x = 0.0      # 前后速度
        self.vel_y = 0.0      # 左右速度
        self.vel_yaw = 0.0    # 旋转速度
        self.vel_magnitude = 0.8  # 恒定速度大小
        
        # 当前技能命令（仅在按键释放时触发一次）
        self.current_skill_cmd = None
        
        # 按键状态记录
        self.key_states = {
            'w': False, 's': False,
            'a': False, 'd': False,
            'q': False, 'e': False,
        }
        
        # 技能按键记录（用于检测释放）
        self.skill_key_pressed = {
            '0': False, '1': False, '2': False, '3': False,
            '4': False, '5': False, '6': False, '7': False, '8': False, '9': False,
        }
        
        # 启动键盘监听器
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self.listener.start()
        
        self._print_help()
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print("实时键盘控制器已启动 - 按下即动，松开即停")
        print("="*60)
        print("技能命令（按下后释放触发）：")
        print("  0 - 被动模式 (PASSIVE)")
        print("  1 - 位置重置 (POS_RESET)")
        print("  2 - 站起来 (STAND_UP)")
        print("  3 - 运动模式 (LOCO) ⭐")
        print("  4 - 技能表演 (SKILL_CAST)")
        print("  5 - 舞蹈模式 (SKILL_Dance)")
        print("  6 - (已删除)")
        print("  7 - (已删除)")
        print("  8 - (已删除)")
        print("  9 - 动作跟踪 (MotionTracking)")
        print("\n移动控制（按住生效，松开归零）：")
        print("  W - 前进 (按住)")
        print("  S - 后退 (按住)")
        print("  A - 左移 (按住)")
        print("  D - 右移 (按住)")
        print("  Q - 左转 (按住)")
        print("  E - 右转 (按住)")
        print("\n其他：")
        print("  H - 显示帮助")
        print("  ESC - 退出程序")
        print("="*60)
        print("提示: 先按 '3' 进入运动模式，然后按住 WASD/QE 控制移动")
        print("="*60 + "\n")
    
    def _on_key_press(self, key):
        """按键按下事件处理"""
        try:
            # 获取按键字符
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                
                # 移动控制 - 按下时设置速度
                if char == 'w':
                    self.key_states['w'] = True
                    self.vel_x = self.vel_magnitude
                    print(f"→ 前进 (速度: {self.vel_x:.2f})")
                elif char == 's':
                    self.key_states['s'] = True
                    self.vel_x = -self.vel_magnitude
                    print(f"→ 后退 (速度: {self.vel_x:.2f})")
                elif char == 'a':
                    self.key_states['a'] = True
                    self.vel_y = self.vel_magnitude
                    print(f"→ 左移 (速度: {self.vel_y:.2f})")
                elif char == 'd':
                    self.key_states['d'] = True
                    self.vel_y = -self.vel_magnitude
                    print(f"→ 右移 (速度: {self.vel_y:.2f})")
                elif char == 'q':
                    self.key_states['q'] = True
                    self.vel_yaw = self.vel_magnitude
                    print(f"→ 左转 (速度: {self.vel_yaw:.2f})")
                elif char == 'e':
                    self.key_states['e'] = True
                    self.vel_yaw = -self.vel_magnitude
                    print(f"→ 右转 (速度: {self.vel_yaw:.2f})")
                
                # 技能按键 - 记录按下状态
                elif char in self.skill_key_pressed:
                    self.skill_key_pressed[char] = True
                
                # 帮助
                elif char == 'h':
                    self._print_help()
                    
            # ESC键退出
            elif key == keyboard.Key.esc:
                print("\n收到ESC退出信号")
                self.current_skill_cmd = CommandType.EXIT
                self.running = False
                return False  # 停止监听器
                
        except Exception as e:
            pass
    
    def _on_key_release(self, key):
        """按键释放事件处理"""
        try:
            # 获取按键字符
            if hasattr(key, 'char') and key.char:
                char = key.char.lower()
                
                # 移动控制 - 释放时归零速度
                if char == 'w' and self.key_states['w']:
                    self.key_states['w'] = False
                    if not self.key_states['s']:  # 如果S没按下
                        self.vel_x = 0.0
                        print("○ 前后停止")
                elif char == 's' and self.key_states['s']:
                    self.key_states['s'] = False
                    if not self.key_states['w']:  # 如果W没按下
                        self.vel_x = 0.0
                        print("○ 前后停止")
                elif char == 'a' and self.key_states['a']:
                    self.key_states['a'] = False
                    if not self.key_states['d']:  # 如果D没按下
                        self.vel_y = 0.0
                        print("○ 左右停止")
                elif char == 'd' and self.key_states['d']:
                    self.key_states['d'] = False
                    if not self.key_states['a']:  # 如果A没按下
                        self.vel_y = 0.0
                        print("○ 左右停止")
                elif char == 'q' and self.key_states['q']:
                    self.key_states['q'] = False
                    if not self.key_states['e']:  # 如果E没按下
                        self.vel_yaw = 0.0
                        print("○ 旋转停止")
                elif char == 'e' and self.key_states['e']:
                    self.key_states['e'] = False
                    if not self.key_states['q']:  # 如果Q没按下
                        self.vel_yaw = 0.0
                        print("○ 旋转停止")
                
                # 技能命令 - 释放时触发
                elif char in self.skill_key_pressed and self.skill_key_pressed[char]:
                    self.skill_key_pressed[char] = False
                    skill_num = int(char)
                    
                    if skill_num == 0:
                        print("→ 切换到被动模式")
                        self.current_skill_cmd = CommandType.PASSIVE
                    elif skill_num == 1:
                        print("→ 执行位置重置")
                        self.current_skill_cmd = CommandType.POS_RESET
                    elif skill_num == 2:
                        print("→ 执行站起来")
                        self.current_skill_cmd = CommandType.STAND_UP
                    elif skill_num == 3:
                        print("→ 切换到运动模式")
                        self.current_skill_cmd = CommandType.LOCO
                    elif skill_num == 4:
                        print("→ 执行技能表演 (SKILL_CAST)")
                        self.current_skill_cmd = CommandType.SKILL_1
                    elif skill_num == 5:
                        print("→ 执行舞蹈模式 (SKILL_Dance)")
                        self.current_skill_cmd = CommandType.SKILL_2
                    elif skill_num == 6:
                        print("→ 技能6已删除，无效果")
                        # self.current_skill_cmd = CommandType.SKILL_3  # 已删除
                    elif skill_num == 7:
                        print("→ 技能7已删除，无效果")
                        # self.current_skill_cmd = CommandType.SKILL_4  # 已删除
                    elif skill_num == 8:
                        print("→ 技能8已删除，无效果")
                        # self.current_skill_cmd = CommandType.SKILL_5  # 已删除
                    elif skill_num == 9:
                        print("→ 执行动作跟踪 (MotionTracking)")
                        self.current_skill_cmd = CommandType.MOTION_TRACKING
                        
        except Exception as e:
            pass
    
    def update(self):
        """更新控制器状态"""
        # 技能命令在获取后自动清空
        pass
    
    def get_skill_command(self):
        """获取当前技能命令（获取后自动清空）"""
        cmd = self.current_skill_cmd
        self.current_skill_cmd = None  # 获取后立即清空
        return cmd
    
    def get_velocity_command(self):
        """获取速度命令 (x, y, yaw)"""
        return (self.vel_x, self.vel_y, self.vel_yaw)
    
    def is_running(self):
        """检查控制器是否还在运行"""
        return self.running
    
    def quit(self):
        """清理资源"""
        self.running = False
        if self.listener:
            self.listener.stop()
        print("\n实时键盘控制器已关闭")

