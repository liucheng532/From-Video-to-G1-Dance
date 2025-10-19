import pygame
from pygame.locals import *
from enum import IntEnum, unique
import time

@unique
class KeyboardButton(IntEnum):
    """键盘按键映射到手柄按钮的枚举"""
    # 基础按键映射
    A = 0      # 空格键 (Space)
    B = 1      # ESC键
    X = 2      # X键
    Y = 3      # Y键
    L1 = 4     # Q键
    R1 = 5     # E键
    SELECT = 6 # Tab键
    START = 7  # Enter键
    L3 = 8     # 左Shift键
    R3 = 9     # 右Shift键
    HOME = 10  # H键
    UP = 11    # W键
    DOWN = 12  # S键
    LEFT = 13  # A键
    RIGHT = 14 # D键

class KeyboardController:
    """键盘控制器类，提供与手柄相同的接口"""
    
    def __init__(self):
        pygame.init()
        
        # 键盘按键映射到手柄按钮
        self.key_mapping = {
            K_SPACE: KeyboardButton.A,      # 空格 -> A
            K_ESCAPE: KeyboardButton.B,     # ESC -> B
            K_x: KeyboardButton.X,          # X -> X
            K_y: KeyboardButton.Y,          # Y -> Y
            K_q: KeyboardButton.L1,         # Q -> L1
            K_e: KeyboardButton.R1,         # E -> R1
            K_TAB: KeyboardButton.SELECT,   # Tab -> SELECT
            K_RETURN: KeyboardButton.START, # Enter -> START
            K_LSHIFT: KeyboardButton.L3,    # 左Shift -> L3
            K_RSHIFT: KeyboardButton.R3,    # 右Shift -> R3
            K_h: KeyboardButton.HOME,       # H -> HOME
            K_w: KeyboardButton.UP,         # W -> UP
            K_s: KeyboardButton.DOWN,       # S -> DOWN
            K_a: KeyboardButton.LEFT,       # A -> LEFT
            K_d: KeyboardButton.RIGHT,      # D -> RIGHT
        }
        
        # 摇杆轴映射 (WASD + 方向键)
        self.axis_mapping = {
            'left_x': [K_a, K_d],    # A/D键控制左摇杆X轴
            'left_y': [K_w, K_s],    # W/S键控制左摇杆Y轴
            'right_x': [K_LEFT, K_RIGHT],  # 左右方向键控制右摇杆X轴
            'right_y': [K_UP, K_DOWN],     # 上下方向键控制右摇杆Y轴
        }
        
        # 按钮状态
        self.button_count = len(KeyboardButton)
        self.button_states = [False] * self.button_count
        self.button_pressed = [False] * self.button_count
        self.button_released = [False] * self.button_count
        
        # 轴状态 (模拟摇杆)
        self.axis_count = 4  # 左X, 左Y, 右X, 右Y
        self.axis_states = [0.0] * self.axis_count
        
        # 帽子状态 (十字键)
        self.hat_count = 1
        self.hat_states = [(0, 0)] * self.hat_count
        
        print("键盘控制器已初始化")
        print("按键映射:")
        print("  空格=A, ESC=B, X=X, Y=Y")
        print("  Q=L1, E=R1, Tab=SELECT, Enter=START")
        print("  左Shift=L3, 右Shift=R3, H=HOME")
        print("  WASD=左摇杆, 方向键=右摇杆")
        
    def update(self):
        """更新键盘状态"""
        pygame.event.pump()
        
        # 重置按钮释放状态
        self.button_released = [False] * self.button_count
        
        # 获取当前按键状态
        keys = pygame.key.get_pressed()
        
        # 更新按钮状态
        for key, button in self.key_mapping.items():
            current_state = keys[key]
            if self.button_states[button] and not current_state:
                self.button_released[button] = True
            self.button_states[button] = current_state
        
        # 更新轴状态 (模拟摇杆)
        # 左摇杆 X轴 (A/D)
        if keys[K_a] and not keys[K_d]:
            self.axis_states[0] = -1.0  # 左
        elif keys[K_d] and not keys[K_a]:
            self.axis_states[0] = 1.0   # 右
        else:
            self.axis_states[0] = 0.0   # 中
            
        # 左摇杆 Y轴 (W/S)
        if keys[K_w] and not keys[K_s]:
            self.axis_states[1] = -1.0  # 上
        elif keys[K_s] and not keys[K_w]:
            self.axis_states[1] = 1.0   # 下
        else:
            self.axis_states[1] = 0.0   # 中
            
        # 右摇杆 X轴 (左右方向键)
        if keys[K_LEFT] and not keys[K_RIGHT]:
            self.axis_states[2] = -1.0  # 左
        elif keys[K_RIGHT] and not keys[K_LEFT]:
            self.axis_states[2] = 1.0   # 右
        else:
            self.axis_states[2] = 0.0   # 中
            
        # 右摇杆 Y轴 (上下方向键)
        if keys[K_UP] and not keys[K_DOWN]:
            self.axis_states[3] = -1.0  # 上
        elif keys[K_DOWN] and not keys[K_UP]:
            self.axis_states[3] = 1.0   # 下
        else:
            self.axis_states[3] = 0.0   # 中
        
        # 更新帽子状态 (十字键)
        hat_x = 0
        hat_y = 0
        if keys[K_LEFT]:
            hat_x = -1
        elif keys[K_RIGHT]:
            hat_x = 1
        if keys[K_UP]:
            hat_y = 1
        elif keys[K_DOWN]:
            hat_y = -1
        self.hat_states[0] = (hat_x, hat_y)
    
    def is_button_pressed(self, button_id):
        """检测按钮是否按下"""
        if 0 <= button_id < self.button_count:
            return self.button_states[button_id]
        return False
    
    def is_button_released(self, button_id):
        """检测按钮是否释放"""
        if 0 <= button_id < self.button_count:
            return self.button_released[button_id]
        return False
    
    def get_axis_value(self, axis_id):
        """获取摇杆轴值"""
        if 0 <= axis_id < self.axis_count:
            return self.axis_states[axis_id]
        return 0.0
    
    def get_hat_direction(self, hat_id=0):
        """获取十字键方向"""
        if 0 <= hat_id < self.hat_count:
            return self.hat_states[hat_id]
        return (0, 0)
    
    def quit(self):
        """清理资源"""
        pygame.quit()

