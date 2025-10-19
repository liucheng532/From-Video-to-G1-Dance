"""
命令行输入控制器 - 使用独立线程接收命令行指令
"""
import threading
import queue
from enum import IntEnum
import sys
import select
import termios
import tty

class CommandType(IntEnum):
    """命令类型枚举"""
    PASSIVE = 0        # 被动模式
    POS_RESET = 1      # 位置重置
    STAND_UP = 2       # 站起来
    LOCO = 3           # 运动模式
    SKILL_1 = 4        # 技能1 (舞蹈)
    SKILL_2 = 5        # 技能2 (功夫2)
    # SKILL_3, SKILL_4, SKILL_5 已删除
    VEL_FORWARD = 10   # 前进
    VEL_BACKWARD = 11  # 后退
    VEL_LEFT = 12      # 左移
    VEL_RIGHT = 13     # 右移
    VEL_ROTATE_LEFT = 14   # 左转
    VEL_ROTATE_RIGHT = 15  # 右转
    VEL_STOP = 16      # 停止移动
    EXIT = 99          # 退出

class CommandInputController:
    """命令行输入控制器"""
    
    def __init__(self):
        """初始化命令行控制器"""
        self.command_queue = queue.Queue()
        self.running = True
        
        # 速度控制参数
        self.vel_x = 0.0  # 前后速度
        self.vel_y = 0.0  # 左右速度
        self.vel_yaw = 0.0  # 旋转速度
        self.vel_step = 0.3  # 速度增量
        self.max_vel = 1.0   # 最大速度
        
        # 当前技能命令
        self.current_skill_cmd = None
        
        # 启动输入线程
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
        
        self._print_help()
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n" + "="*60)
        print("命令行控制器已启动")
        print("="*60)
        print("技能命令：")
        print("  0 - 被动模式 (PASSIVE)")
        print("  1 - 位置重置 (POS_RESET)")
        print("  2 - 站起来 (STAND_UP)")
        print("  3 - 运动模式 (LOCO)")
        print("  4 - 技能1 (舞蹈)")
        print("  5 - 技能2 (功夫2)")
        print("  6 - 技能3 (跳舞)")
        print("  7 - 技能4 (功夫1)")
        print("  8 - 技能5 (ASAP)")
        print("\n移动控制 (在LOCO模式下)：")
        print("  w - 前进")
        print("  s - 后退")
        print("  a - 左移")
        print("  d - 右移")
        print("  q - 左转")
        print("  e - 右转")
        print("  x - 停止移动")
        print("\n其他：")
        print("  h - 显示帮助")
        print("  exit/quit/q - 退出程序")
        print("="*60)
        print("请输入命令: ", end='', flush=True)
    
    def _input_loop(self):
        """输入循环，在独立线程中运行"""
        while self.running:
            try:
                # 读取一行输入
                cmd = input().strip().lower()
                
                if cmd:
                    self.command_queue.put(cmd)
                    print("请输入命令: ", end='', flush=True)
                    
            except (EOFError, KeyboardInterrupt):
                self.running = False
                break
            except Exception as e:
                print(f"\n输入错误: {e}")
                print("请输入命令: ", end='', flush=True)
    
    def update(self):
        """更新控制器状态，处理命令队列"""
        # 重置技能命令
        self.current_skill_cmd = None
        
        # 处理所有待处理的命令
        while not self.command_queue.empty():
            try:
                cmd = self.command_queue.get_nowait()
                self._process_command(cmd)
            except queue.Empty:
                break
    
    def _process_command(self, cmd):
        """处理单个命令"""
        # 退出命令
        if cmd in ['exit', 'quit', 'q']:
            print("\n收到退出命令")
            self.current_skill_cmd = CommandType.EXIT
            self.running = False
            return
        
        # 帮助命令
        if cmd == 'h' or cmd == 'help':
            self._print_help()
            return
        
        # 技能命令 (数字)
        if cmd.isdigit():
            skill_num = int(cmd)
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
                print("→ 执行技能1 (舞蹈)")
                self.current_skill_cmd = CommandType.SKILL_1
            elif skill_num == 5:
                print("→ 执行技能2 (功夫2)")
                self.current_skill_cmd = CommandType.SKILL_2
            elif skill_num == 6:
                print("→ 技能6已删除，无效果")
            elif skill_num == 7:
                print("→ 技能7已删除，无效果")
            elif skill_num == 8:
                print("→ 技能8已删除，无效果")
            else:
                print(f"✗ 未知技能编号: {skill_num}")
            return
        
        # 移动控制命令
        if cmd == 'w':
            self.vel_x = min(self.vel_x + self.vel_step, self.max_vel)
            print(f"→ 前进 (速度: {self.vel_x:.2f})")
        elif cmd == 's':
            self.vel_x = max(self.vel_x - self.vel_step, -self.max_vel)
            print(f"→ 后退 (速度: {self.vel_x:.2f})")
        elif cmd == 'a':
            self.vel_y = min(self.vel_y + self.vel_step, self.max_vel)
            print(f"→ 左移 (速度: {self.vel_y:.2f})")
        elif cmd == 'd':
            self.vel_y = max(self.vel_y - self.vel_step, -self.max_vel)
            print(f"→ 右移 (速度: {self.vel_y:.2f})")
        elif cmd == 'q':
            self.vel_yaw = min(self.vel_yaw + self.vel_step, self.max_vel)
            print(f"→ 左转 (速度: {self.vel_yaw:.2f})")
        elif cmd == 'e':
            self.vel_yaw = max(self.vel_yaw - self.vel_step, -self.max_vel)
            print(f"→ 右转 (速度: {self.vel_yaw:.2f})")
        elif cmd == 'x':
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.vel_yaw = 0.0
            print("→ 停止移动")
        else:
            print(f"✗ 未知命令: {cmd}")
            print("  输入 'h' 查看帮助")
    
    def get_skill_command(self):
        """获取当前技能命令"""
        return self.current_skill_cmd
    
    def get_velocity_command(self):
        """获取速度命令 (x, y, yaw)"""
        return (self.vel_x, self.vel_y, self.vel_yaw)
    
    def is_running(self):
        """检查控制器是否还在运行"""
        return self.running
    
    def quit(self):
        """清理资源"""
        self.running = False
        if self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        print("\n命令行控制器已关闭")


