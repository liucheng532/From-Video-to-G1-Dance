import pygame
from .joystick import JoyStick, JoystickButton
from .keyboard_controller import KeyboardController, KeyboardButton
from enum import IntEnum, unique

@unique
class ControllerType(IntEnum):
    """控制器类型枚举"""
    JOYSTICK = 0
    KEYBOARD = 1

class UnifiedController:
    """统一控制器类，自动检测手柄或键盘"""
    
    def __init__(self, prefer_joystick=True):
        """
        初始化统一控制器
        
        Args:
            prefer_joystick (bool): 是否优先使用手柄，如果为False则优先使用键盘
        """
        pygame.init()
        pygame.joystick.init()
        
        self.controller_type = None
        self.controller = None
        self.prefer_joystick = prefer_joystick
        
        # 尝试初始化控制器
        self._initialize_controller()
        
    def _initialize_controller(self):
        """初始化控制器，优先手柄，其次键盘"""
        if self.prefer_joystick:
            # 优先尝试手柄
            if self._try_joystick():
                return
            # 手柄失败，尝试键盘
            if self._try_keyboard():
                return
        else:
            # 优先尝试键盘
            if self._try_keyboard():
                return
            # 键盘失败，尝试手柄
            if self._try_joystick():
                return
        
        # 如果都失败了，抛出异常
        raise RuntimeError("无法初始化任何控制器！请连接手柄或确保键盘可用。")
    
    def _try_joystick(self):
        """尝试初始化手柄"""
        try:
            joystick_count = pygame.joystick.get_count()
            if joystick_count > 0:
                self.controller = JoyStick()
                self.controller_type = ControllerType.JOYSTICK
                print(f"✓ 手柄控制器已连接: {self.controller.joystick.get_name()}")
                return True
        except Exception as e:
            print(f"手柄初始化失败: {e}")
        return False
    
    def _try_keyboard(self):
        """尝试初始化键盘"""
        try:
            self.controller = KeyboardController()
            self.controller_type = ControllerType.KEYBOARD
            print("✓ 键盘控制器已激活")
            return True
        except Exception as e:
            print(f"键盘初始化失败: {e}")
        return False
    
    def update(self):
        """更新控制器状态"""
        if self.controller:
            self.controller.update()
    
    def is_button_pressed(self, button_id):
        """检测按钮是否按下"""
        if self.controller:
            return self.controller.is_button_pressed(button_id)
        return False
    
    def is_button_released(self, button_id):
        """检测按钮是否释放"""
        if self.controller:
            return self.controller.is_button_released(button_id)
        return False
    
    def get_axis_value(self, axis_id):
        """获取摇杆轴值"""
        if self.controller:
            return self.controller.get_axis_value(axis_id)
        return 0.0
    
    def get_hat_direction(self, hat_id=0):
        """获取十字键方向"""
        if self.controller:
            return self.controller.get_hat_direction(hat_id)
        return (0, 0)
    
    def get_controller_type(self):
        """获取当前控制器类型"""
        return self.controller_type
    
    def get_controller_name(self):
        """获取控制器名称"""
        if self.controller_type == ControllerType.JOYSTICK:
            return f"手柄: {self.controller.joystick.get_name()}"
        elif self.controller_type == ControllerType.KEYBOARD:
            return "键盘控制器"
        return "未知控制器"
    
    def switch_controller(self):
        """切换控制器类型"""
        if self.controller_type == ControllerType.JOYSTICK:
            # 从手柄切换到键盘
            if self._try_keyboard():
                print("已切换到键盘控制器")
                return True
        elif self.controller_type == ControllerType.KEYBOARD:
            # 从键盘切换到手柄
            if self._try_joystick():
                print("已切换到手柄控制器")
                return True
        
        print("控制器切换失败")
        return False
    
    def quit(self):
        """清理资源"""
        if self.controller:
            if hasattr(self.controller, 'quit'):
                self.controller.quit()
        pygame.quit()

# 为了保持向后兼容性，创建一个别名
# 这样现有的代码可以继续使用 JoystickButton
class UnifiedButton(IntEnum):
    """统一按钮枚举，兼容手柄和键盘"""
    # 基础按钮
    A = 0
    B = 1
    X = 2
    Y = 3
    L1 = 4
    R1 = 5
    SELECT = 6
    START = 7
    L3 = 8
    R3 = 9
    HOME = 10
    UP = 11
    DOWN = 12
    LEFT = 13
    RIGHT = 14

# 为了完全兼容，将 JoystickButton 映射到 UnifiedButton
# 这样现有代码不需要修改
JoystickButton = UnifiedButton

