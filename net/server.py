# -*- coding: utf-8 -*-
# @Author  : Amilime
# @Time    : 11/1/2025 1:08 PM
# @File    : server.py
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
    # 从yolopeople导入新的运动检测器
    from yolopeople import LocalMotionMonitor  # 使用局部运动检测版本

    IMPORT_SUCCESS = True
    print("YOLO人员运动检测模块导入成功")
except ImportError as e:
    print(f"导入YOLO运动检测模块失败: {e}")
    print("请确保yolopeople.py在正确的路径")
    IMPORT_SUCCESS = False


class StepperServer:
    def __init__(self, host='0.0.0.0', port=8888, import_success=False):  # 改为0.0.0.0监听所有接口
        self.host = host
        self.port = port
        self.server_socket = None
        self.is_running = False
        self.clients = []
        self.speed = 0  # 默认速度
        self.mode = "STOP"  # 控制模式: STOP, RUN
        self.send_interval = 0.1

        # 初始化屏幕运动检测
        self.motion_monitor = None
        self.current_motion_intensity = 0.0
        self.IMPORT_SUCCESS = import_success

        if self.IMPORT_SUCCESS:
            try:
                # 使用绝对路径避免下载
                model_path = os.path.join(cv_dir, 'yolo11s.pt')
                print(f"尝试加载模型: {model_path}")

                if os.path.exists(model_path):
                    self.motion_monitor = LocalMotionMonitor(
                        model_path=model_path,
                        conf_threshold=0.3
                    )
                    print("YOLO人员运动检测模块初始化成功")
                else:
                    print(f"模型文件不存在: {model_path}")
                    self.IMPORT_SUCCESS = False
            except Exception as e:
                print(f"YOLO运动检测初始化失败: {e}")
                self.IMPORT_SUCCESS = False
        else:
            print("将使用模拟运动数据")

    def set_speed_from_motion(self, motion_intensity):
        """从屏幕运动检测程序接收运动权重值"""
        # 存储当前运动权重
        self.current_motion_intensity = motion_intensity

        # 只在RUN模式下计算速度
        if self.mode == "RUN":
            # 将运动权重映射到速度值 (0-255)
            if motion_intensity < 0.05:
                normalized = motion_intensity / 0.05 * 0.2
            elif motion_intensity < 0.15:
                normalized = 0.2 + (motion_intensity - 0.05) / 0.1 * 0.4
            elif motion_intensity < 0.3:
                normalized = 0.6 + (motion_intensity - 0.15) / 0.15 * 0.3
            else:
                normalized = 0.9 + (motion_intensity - 0.3) / 0.7 * 0.1

            speed_value = int(normalized * 255)
            self.speed = min(max(speed_value, 0), 255)
        else:
            # STOP模式下速度保持为0
            self.speed = 0

        return self.speed

    def start_motion_detection(self):
        """启动屏幕运动检测 - 修复：只在RUN模式下进行检测计算"""
        print(f"DEBUG: 准备启动运动检测线程, IMPORT_SUCCESS={self.IMPORT_SUCCESS}")

        def motion_detection_loop():
            print(f"DEBUG: 运动检测线程开始运行")

            if not self.IMPORT_SUCCESS or not self.motion_monitor:
                print("使用模拟运动数据")
                self.simulate_motion_data()
                return

            print("启动真实的屏幕运动检测...")
            frame_count = 0
            last_print_time = time.time()
            last_detection_time = 0
            detection_interval = 0.5  # 降低检测频率，减少CPU占用

            while self.is_running:
                frame_count += 1
                current_time = time.time()

                try:
                    # 只在RUN模式下进行密集检测，STOP模式下降低频率
                    if self.mode == "RUN":
                        if current_time - last_detection_time >= detection_interval:
                            # 获取屏幕
                            frame = self.motion_monitor.get_screen()
                            if frame is None:
                                time.sleep(0.1)
                                continue

                            # 检测人员
                            persons = self.motion_monitor.detect_persons_with_pose(frame)

                            # 计算局部运动
                            local_motion = self.motion_monitor.compute_local_motion(persons, frame)

                            # 计算帧间运动
                            frame_motion = self.motion_monitor.compute_frame_motion(frame)

                            # 综合运动指标
                            combined_motion = local_motion * 0.7 + frame_motion * 0.3

                            # 更新速度
                            speed = self.set_speed_from_motion(combined_motion)
                            level, _ = self.motion_monitor.get_motion_level(combined_motion)

                            # 每秒输出一次详细信息
                            if current_time - last_print_time >= 1.0:
                                print(
                                    f"YOLO检测: 强度{combined_motion:.4f} -> 速度{speed:3d} | {level} | 人数:{len(persons)}")
                                last_print_time = current_time

                            last_detection_time = current_time
                    else:
                        # STOP模式下每5秒检测一次，仅用于状态显示
                        if current_time - last_print_time >= 5.0:
                            frame = self.motion_monitor.get_screen()
                            if frame is not None:
                                persons = self.motion_monitor.detect_persons_with_pose(frame)
                                print(f"STOP模式检测: 人数:{len(persons)} | 客户端:{len(self.clients)}")
                                last_print_time = current_time

                    time.sleep(0.1)  # 基础休眠时间

                except Exception as e:
                    print(f"运动检测错误: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)

        motion_thread = threading.Thread(target=motion_detection_loop)
        motion_thread.daemon = True
        motion_thread.start()
        print(f"DEBUG: 运动检测线程已启动")

    def simulate_motion_data(self):
        """模拟运动数据（备用方案）- 修复：只在RUN模式下模拟"""
        import random
        simulation_count = 0
        last_print_time = 0

        while self.is_running:
            try:
                current_time = time.time()

                # 只在RUN模式下生成模拟速度
                if self.mode == "RUN":
                    # 模拟运动权重 (0-10)
                    if simulation_count % 50 < 25:
                        motion_intensity = (simulation_count % 25) / 25.0 * 0.3
                    else:
                        motion_intensity = random.uniform(0, 0.4)

                    speed = self.set_speed_from_motion(motion_intensity)

                    simulation_count += 1
                    if current_time - last_print_time >= 1.0:
                        level = "静止" if motion_intensity < 0.05 else "微动" if motion_intensity < 0.1 else "运动" if motion_intensity < 0.2 else "剧烈"
                        print(
                            f"模拟运动: 强度{motion_intensity:.4f} -> 速度{speed:3d} | {level} | 客户端:{len(self.clients)}")
                        last_print_time = current_time
                else:
                    # STOP模式下每5秒显示一次状态
                    if current_time - last_print_time >= 5.0:
                        print(f"STOP模式: 速度{self.speed} | 客户端:{len(self.clients)}")
                        last_print_time = current_time
                        simulation_count = 0  # 重置计数

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
            self.server_socket.settimeout(1.0)

            # 获取本机IP地址用于显示
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
                s.close()
            except:
                local_ip = "127.0.0.1"

            print(f"步进电机控制服务器启动成功!")
            print(f"监听地址: {self.host}:{self.port}")
            print(f"客户端连接地址: {local_ip}:{self.port}")
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
                print(f"\n✅ 单片机连接成功: {client_address[0]}:{client_address[1]}")
                self.clients.append(client_socket)

            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"接受连接时出错: {e}")

    def send_speed_data(self):
        """持续向单片机发送速度数据 - 修复：发送格式与ESP32客户端匹配"""
        last_speed = -1

        while self.is_running:
            try:
                if self.clients:
                    # 发送数据格式: "速度值\n" (ESP32期望的格式)
                    speed_str = f"{self.speed}\n"

                    # 只在速度变化时发送，减少网络负载
                    if self.speed != last_speed:
                        disconnected_clients = []
                        for client_socket in self.clients:
                            try:
                                client_socket.send(speed_str.encode('utf-8'))
                                # 调试输出
                                if self.mode == "RUN" and self.speed != last_speed:
                                    print(f"发送速度: {self.speed}")
                            except Exception as e:
                                disconnected_clients.append(client_socket)

                        # 移除断开的客户端
                        for client in disconnected_clients:
                            self.clients.remove(client)
                            print("❌ 客户端断开连接")

                        last_speed = self.speed

                time.sleep(self.send_interval)

            except Exception as e:
                print(f"发送数据时出错: {e}")
                time.sleep(1)

    def set_mode(self, mode):
        """设置控制模式 - 修复：重置速度计算"""
        if mode in ["STOP", "RUN"]:
            old_mode = self.mode
            self.mode = mode

            if mode == "STOP":
                self.speed = 0
                if old_mode == "RUN":
                    print(f"\n🛑 控制模式: RUN -> STOP (速度归零)")
                    # 重置运动检测器
                    if self.motion_monitor:
                        self.motion_monitor.prev_person_boxes = None
                        self.motion_monitor.prev_frame = None
            else:
                if old_mode == "STOP":
                    print(f"\n🎯 控制模式: STOP -> RUN (开始发送速度数据)")
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
                    print(f"运动强度: {self.current_motion_intensity:.4f}")
                    print(f"连接客户端: {len(self.clients)}")
                    print(f"屏幕检测: {'运行中' if self.IMPORT_SUCCESS else '模拟模式'}")
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

        for client in self.clients:
            try:
                client.close()
            except:
                pass
        self.clients.clear()

        print("服务器已完全停止")


# 使用示例
if __name__ == "__main__":
    print(f"IMPORT_SUCCESS: {IMPORT_SUCCESS}")

    # 创建服务器实例，使用0.0.0.0监听所有网络接口
    server = StepperServer('0.0.0.0', 8888, import_success=IMPORT_SUCCESS)

    # 启动服务器
    server.start_server()
