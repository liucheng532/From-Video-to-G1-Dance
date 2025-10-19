import pygame
import time

def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("未检测到手柄，请连接后重试。")
        return

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"检测到手柄: {joystick.get_name()}")
    print(f"按钮数: {joystick.get_numbuttons()}")
    print(f"轴数（摇杆轴）: {joystick.get_numaxes()}")
    print(f"帽子开关数（十字键）: {joystick.get_numhats()}")
    print("开始监测手柄输入...（按 Ctrl+C 退出）")

    # 记录初始轴状态
    axis_baseline = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]

    try:
        while True:
            pygame.event.pump()

            # 按钮检测
            for i in range(joystick.get_numbuttons()):
                if joystick.get_button(i):
                    print(f"[按钮] 按下: ID={i}")

            # 轴检测（摇杆 + 扳机）
            for i in range(joystick.get_numaxes()):
                val = joystick.get_axis(i)
                baseline = axis_baseline[i]
                if abs(val - baseline) > 0.2:  # 加大阈值过滤静态轴
                    print(f"[摇杆/扳机] 轴 ID={i}, 当前值={val:.2f}, 初始值={baseline:.2f}")

            # 十字键检测
            for i in range(joystick.get_numhats()):
                hat = joystick.get_hat(i)
                if hat != (0, 0):
                    print(f"[D-Pad] 方向: ID={i}, 值={hat}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n退出手柄监测。")
    finally:
        joystick.quit()
        pygame.quit()

if __name__ == "__main__":
    main()
