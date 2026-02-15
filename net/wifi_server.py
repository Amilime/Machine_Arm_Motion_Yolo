# -*- coding: utf-8 -*-
# @Author  : Amilime
# @Time    : 10/31/2025 11:06 PM
# @File    : wifi_server.py
# @Software: PyCharm

import socket
import threading
import time
import sys
import os

# 添加cv文件夹到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
cv_dir = os.path.join(current_dir, '..', 'cv')
sys.path.append(cv_dir)

try:
    from screen_catch import MotionIntensityDetector, ScreenCapture

    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"导入屏幕捕获模块失败: {e}")
    print("请确保screen_catch.py在正确的路径")
    IMPORT_SUCCESS = False


class StepperServer:
    def __init__(self, host='localhost', port=8888, import_success=False):
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = False
        self.clients = []
        self.speed = 0  # 默认速度
        self.mode = "STOP"  # 控制模式: STOP, RUN
        self.send_interval = 1.0  # 发送间隔(秒)，降低到0.5秒提高响应速度

        # 初始化屏幕运动检测
        self.motion_detector = None
        self.screen_capture = None
        self.current_motion_weight = 0.0
        self.IMPORT_SUCCESS = import_success

        if self.IMPORT_SUCCESS:
            try:
                self.motion_detector = MotionIntensityDetector(
                    window_size=15,
                    intensity_threshold=15,
                    roi=None
                )
                self.screen_capture = ScreenCapture(reduce_scale=0.4)
                print("屏幕运动检测模块初始化成功")
            except Exception as e:
                print(f"运动检测初始化失败: {e}")
                self.IMPORT_SUCCESS = False  # 这里要改为 self.IMPORT_SUCCESS
        else:
            print("将使用模拟运动数据")

    def set_speed_from_motion(self, motion_weight):
        """从屏幕运动检测程序接收运动权重值"""
        # 存储当前运动权重
        self.current_motion_weight = motion_weight

        # 将运动权重映射到速度值 (0-255)
        # motion_weight 范围是 0-10，我们将其映射到 0-255
        # 使用非线性映射：低运动时变化平缓，高运动时变化剧烈
        if motion_weight < 2:
            # 低运动区域：平缓增长
            normalized = motion_weight / 2.0 * 0.3
        elif motion_weight < 5:
            # 中运动区域：线性增长
            normalized = 0.3 + (motion_weight - 2) / 3.0 * 0.4
        else:
            # 高运动区域：快速增长
            normalized = 0.7 + (motion_weight - 5) / 5.0 * 0.3

        speed_value = int(normalized * 255)
        self.speed = min(max(speed_value, 0), 255)  # 限制在0-255范围内
        return self.speed

    def start_motion_detection(self):
        """启动屏幕运动检测"""
        print(f"DEBUG: 准备启动运动检测线程, IMPORT_SUCCESS={self.IMPORT_SUCCESS}")

        def motion_detection_loop():
            print(f"DEBUG: 运动检测线程开始运行")

            if not self.IMPORT_SUCCESS or not self.motion_detector or not self.screen_capture:
                print("使用模拟运动数据")
                self.simulate_motion_data()
                return

            print("启动真实的屏幕运动检测...")
            detection_count = 0

            while self.is_running:
                # 只有在RUN模式下才进行运动检测和速度计算
                if self.mode != "RUN":
                    # STOP模式下只做最基本的检测，不更新速度
                    try:
                        frame = self.screen_capture.capture_screen()
                        if frame is not None:
                            # 只处理帧但不更新速度，保持检测器状态
                            result = self.motion_detector.process_frame(frame)
                            motion_weight = result['current_weight']

                            detection_count += 1
                            if detection_count % 10 == 0:  # 减少输出频率
                                print(f"STOP模式检测: 权重{motion_weight:.2f} (不更新速度)")

                        time.sleep(0.2)  # STOP模式下检测频率降低
                        continue
                    except Exception as e:
                        time.sleep(0.5)
                        continue

                # RUN模式下的正常检测逻辑
                try:
                    # 捕获屏幕
                    frame = self.screen_capture.capture_screen()
                    if frame is None:
                        time.sleep(0.05)
                        continue

                    # 处理帧并获取运动权重
                    result = self.motion_detector.process_frame(frame)
                    motion_weight = result['current_weight']

                    # 更新速度值
                    speed = self.set_speed_from_motion(motion_weight)

                    detection_count += 1
                    if detection_count % 5 == 0:
                        print(f"RUN模式检测: 权重{motion_weight:.2f} -> 速度{speed:3d} | 客户端:{len(self.clients)}")

                    time.sleep(0.1)

                except Exception as e:
                    print(f"运动检测错误: {e}")
                    time.sleep(0.5)

        motion_thread = threading.Thread(target=motion_detection_loop)
        motion_thread.daemon = True
        motion_thread.start()
        print(f"DEBUG: 运动检测线程已启动，线程ID: {motion_thread.ident}")

    def simulate_motion_data(self):
        """模拟运动数据（备用方案）"""
        import random
        simulation_count = 0

        while self.is_running:
            try:
                # 模拟运动权重 (0-10)
                if simulation_count % 50 < 25:
                    # 模拟周期性运动
                    motion_weight = (simulation_count % 25) / 2.5
                else:
                    # 模拟随机运动
                    motion_weight = random.uniform(0, 8)

                speed = self.set_speed_from_motion(motion_weight)

                simulation_count += 1
                if simulation_count % 20 == 0:
                    print(
                        f"\r模拟运动: 权重{motion_weight:.2f} -> 速度{speed:3d} | 客户端:{len(self.clients)} | 模式:{self.mode}",
                        end="")

                time.sleep(0.1)

            except Exception as e:
                print(f"\n模拟数据错误: {e}")
                time.sleep(1)

    def start_server(self):
        """启动TCP服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # 设置超时以便检查is_running

            print(f"步进电机控制服务器启动成功，监听 {self.host}:{self.port}")
            print("等待单片机连接...")

            self.is_running = True

            # 启动屏幕运动检测
            self.start_motion_detection()

            # 接受客户端连接
            accept_thread = threading.Thread(target=self.accept_clients)
            accept_thread.daemon = True
            accept_thread.start()

            # 持续发送速度数据
            send_thread = threading.Thread(target=self.send_speed_data)
            send_thread.daemon = True
            send_thread.start()

            self.handle_user_input()

        except Exception as e:
            print(f"服务器启动失败: {e}")
        finally:
            self.stop_server()

    def accept_clients(self):
        """接受客户端连接"""
        while self.is_running:
            try:
                client_socket, client_address = self.server_socket.accept()
                client_socket.settimeout(2.0)
                print(f"\n单片机连接成功: {client_address}")
                self.clients.append(client_socket)

            except socket.timeout:
                continue  # 超时是正常的，继续循环
            except Exception as e:
                if self.is_running:
                    print(f"接受连接时出错: {e}")

    def send_speed_data(self):
        """持续向单片机发送速度数据"""
        last_speed = -1  # 记录上次发送的速度

        while self.is_running:
            try:
                if self.clients and self.mode == "RUN":
                    # 只在RUN模式下发送速度数据
                    if self.speed != last_speed:  # 只在速度变化时发送
                        speed_str = f"{self.speed}\n"

                        disconnected_clients = []
                        for client_socket in self.clients:
                            try:
                                client_socket.send(speed_str.encode('utf-8'))
                            except Exception as e:
                                print(f"\n发送失败，客户端可能已断开: {e}")
                                disconnected_clients.append(client_socket)

                        # 移除断开的客户端
                        for client in disconnected_clients:
                            self.clients.remove(client)
                            print("客户端断开连接")

                        last_speed = self.speed

                time.sleep(self.send_interval)

            except Exception as e:
                print(f"发送数据时出错: {e}")
                time.sleep(1)

    def set_mode(self, mode):
        """设置控制模式"""
        if mode in ["STOP", "RUN"]:
            old_mode = self.mode
            self.mode = mode

            if mode == "STOP":
                self.speed = 0  # 停止时速度归零
                if old_mode == "RUN":
                    print(f"\n控制模式: RUN -> STOP (速度归零)")
                    # 重置运动检测器，避免累积效应
                    if self.motion_detector:
                        self.motion_detector.prev_frame = None
            else:
                if old_mode == "STOP":
                    print(f"\n控制模式: STOP -> RUN (开始发送速度数据)")
        else:
            print("无效的模式，请输入 STOP 或 RUN")

    def handle_user_input(self):
        """处理用户输入"""
        print("\n" + "=" * 50)
        print("步进电机控制系统 - 屏幕运动控制版")
        print("=" * 50)
        print("控制命令:")
        print("  run   - 开始发送速度数据")
        print("  stop  - 停止发送速度数据")
        print("  speed - 手动设置速度(0-255)")
        print("  status - 显示当前状态")
        print("  quit  - 退出程序")
        print(f"\n当前模式: {self.mode}, 当前速度: {self.speed}")

        while self.is_running:
            try:
                cmd = input("\n输入命令: ").strip().lower()

                if cmd in ['quit', 'exit']:
                    self.stop_server()
                    break
                elif cmd == 'run':
                    self.set_mode("RUN")
                elif cmd == 'stop':
                    self.set_mode("STOP")
                elif cmd == 'status':
                    print(f"模式: {self.mode}")
                    print(f"速度: {self.speed}")
                    print(f"运动权重: {self.current_motion_weight:.2f}")
                    print(f"连接客户端: {len(self.clients)}")
                    print(f"屏幕检测: {'运行中' if self.IMPORT_SUCCESS else '模拟模式'}")  # 这里改为self.IMPORT_SUCCESS
                elif cmd.startswith('speed '):
                    try:
                        parts = cmd.split()
                        if len(parts) == 2:
                            manual_speed = int(parts[1])
                            if 0 <= manual_speed <= 255:
                                self.speed = manual_speed
                                print(f"手动设置速度: {manual_speed}")
                            else:
                                print("速度值必须在0-255之间")
                    except ValueError:
                        print("请输入有效的数字速度值")
                else:
                    print("未知命令，请输入 run, stop, speed, status 或 quit")

            except KeyboardInterrupt:
                print("\n接收到中断信号...")
                self.stop_server()
                break
            except Exception as e:
                print(f"输入处理错误: {e}")

    def stop_server(self):
        """停止服务器"""
        print("\n正在停止服务器...")
        self.is_running = False
        if self.server_socket:
            self.server_socket.close()

        # 关闭所有客户端连接
        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients.clear()

        print("服务器已完全停止")


# 使用示例
if __name__ == "__main__":
    print(f"IMPORT_SUCCESS: {IMPORT_SUCCESS}")  # 添加调试信息

    # 创建服务器实例，传入IMPORT_SUCCESS状态
    server = StepperServer('0.0.0.0', 8888, import_success=IMPORT_SUCCESS)

    # 启动服务器
    server.start_server()
