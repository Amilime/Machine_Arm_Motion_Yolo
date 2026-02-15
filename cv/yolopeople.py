import numpy as np
import time
from collections import deque
from mss import mss
from ultralytics import YOLO
import cv2


class LocalMotionMonitor:
    def __init__(self, model_path='yolo11s.pt', conf_threshold=0.5):
        print("初始化局部运动监测器...")

        try:
            self.model = YOLO(model_path)
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

        self.conf_threshold = conf_threshold

        # 局部运动检测参数
        self.prev_person_boxes = None
        self.prev_frame = None
        self.motion_log = deque(maxlen=30)
        self.local_motion_log = deque(maxlen=30)

        # 屏幕捕获
        self.sct = mss()
        self.monitor = self.sct.monitors[1]

        # 添加运行控制标志
        self.is_running = False

        print("局部运动监测模式 - 专门检测肢体运动")

    def get_screen(self):
        """获取屏幕图像"""
        try:
            # 检查是否有可用的显示器
            if not hasattr(self, 'sct') or not hasattr(self, 'monitor'):
                print("DEBUG: mss未正确初始化")
                return None

            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)

            if img.size == 0:
                print("DEBUG: 捕获的图像为空")
                return None

            result = img[:, :, :3] if img.shape[2] == 4 else img
            return result

        except Exception as e:
            print(f"DEBUG: 屏幕捕获失败: {e}")
            # 尝试重新初始化mss
            try:
                self.sct = mss()
                self.monitor = self.sct.monitors[1]
                print("DEBUG: 重新初始化mss成功")
            except:
                print("DEBUG: 重新初始化mss失败")
            return None

    def detect_persons_with_pose(self, image):
        """检测人员并获取姿态信息"""
        if image is None:
            return []

        try:
            # 使用pose检测获取更多关键点
            results = self.model(image, verbose=False, conf=self.conf_threshold)
            persons = []

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()

                for i, (box, cls) in enumerate(zip(boxes, classes)):
                    if self.model.names[int(cls)] == 'person':
                        x1, y1, x2, y2 = box
                        # 计算多个关键点位置用于局部运动检测
                        person_data = {
                            'box': box,
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'keypoints': [
                                (x1, y1),  # 左上角
                                (x2, y1),  # 右上角
                                (x1, y2),  # 左下角
                                (x2, y2),  # 右下角
                                ((x1 + x2) / 2, y1),  # 顶部中心
                                ((x1 + x2) / 2, y2),  # 底部中心
                            ]
                        }
                        persons.append(person_data)

            return persons

        except Exception as e:
            print(f"人员检测错误: {e}")
            return []

    def compute_local_motion(self, current_persons, current_frame):
        """计算局部肢体运动"""
        if not current_persons or self.prev_person_boxes is None or self.prev_frame is None:
            self.prev_person_boxes = current_persons
            self.prev_frame = current_frame
            return 0.0

        total_local_motion = 0.0
        valid_persons = 0

        for curr_person in current_persons:
            # 找到最匹配的前一帧人员
            best_match = None
            min_distance = float('inf')

            for prev_person in self.prev_person_boxes:
                distance = np.sqrt(
                    (curr_person['center'][0] - prev_person['center'][0]) ** 2 +
                    (curr_person['center'][1] - prev_person['center'][1]) ** 2
                )
                if distance < min_distance:
                    min_distance = distance
                    best_match = prev_person

            if best_match and min_distance < min(curr_person['width'], curr_person['height']) * 0.5:
                # 计算边界框形状变化（反映肢体运动）
                width_change = abs(curr_person['width'] - best_match['width']) / best_match['width']
                height_change = abs(curr_person['height'] - best_match['height']) / best_match['height']
                aspect_change = abs(
                    (curr_person['width'] / curr_person['height']) -
                    (best_match['width'] / best_match['height'])
                )

                # 计算关键点运动
                keypoint_motion = 0.0
                for curr_kp, prev_kp in zip(curr_person['keypoints'], best_match['keypoints']):
                    kp_distance = np.sqrt(
                        (curr_kp[0] - prev_kp[0]) ** 2 + (curr_kp[1] - prev_kp[1]) ** 2
                    )
                    normalized_kp_motion = kp_distance / min(curr_person['width'], curr_person['height'])
                    keypoint_motion += normalized_kp_motion

                keypoint_motion /= len(curr_person['keypoints'])

                # 综合局部运动指标
                local_motion = (width_change + height_change + aspect_change + keypoint_motion) / 4
                total_local_motion += local_motion
                valid_persons += 1

        self.prev_person_boxes = current_persons
        self.prev_frame = current_frame

        return total_local_motion / valid_persons if valid_persons > 0 else 0.0

    def compute_frame_motion(self, current_frame):
        """计算帧间运动（用于检测整体运动）"""
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return 0.0

        # 转换为灰度图
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # 计算光流或帧差
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

        # 计算运动像素比例
        motion_ratio = np.sum(thresh > 0) / thresh.size

        self.prev_frame = current_frame
        return min(motion_ratio * 20, 1.0)  # 增强局部运动的灵敏度

    def get_motion_level(self, motion_rate):
        """针对局部运动的等级划分"""
        if motion_rate < 0.01:
            return "静止", "\033[90m"
        elif motion_rate < 0.03:
            return "微动", "\033[92m"  # 呼吸、轻微调整
        elif motion_rate < 0.08:
            return "轻度", "\033[93m"  # 打字、手势
        elif motion_rate < 0.15:
            return "中度", "\033[33m"  # 挥手、点头
        elif motion_rate < 0.25:
            return "剧烈", "\033[91m"  # 俯卧撑、跳跃
        else:
            return "极剧烈", "\033[95m"  # 快速运动

    def get_single_detection(self):
        """单次检测 - 供服务器调用"""
        try:
            # 获取屏幕
            frame = self.get_screen()
            if frame is None:
                return 0.0, 0

            # 检测人员
            persons = self.detect_persons_with_pose(frame)

            # 计算局部运动
            local_motion = self.compute_local_motion(persons, frame)

            # 计算帧间运动作为补充
            frame_motion = self.compute_frame_motion(frame)

            # 综合运动指标（侧重局部运动）
            combined_motion = local_motion * 0.7 + frame_motion * 0.3

            return combined_motion, len(persons)

        except Exception as e:
            print(f"单次检测错误: {e}")
            return 0.0, 0

    def run_local_motion_monitor(self):
        """局部运动监测循环 - 独立运行用"""
        print("局部肢体运动监测启动")
        print("专门检测俯卧撑、手势等局部运动")
        print("按 Ctrl+C 停止")
        print("=" * 70)

        self.is_running = True
        last_print_time = time.time()
        print_interval = 0.3

        try:
            while self.is_running:
                # 获取屏幕
                frame = self.get_screen()
                if frame is None:
                    time.sleep(0.02)
                    continue

                # 检测人员
                persons = self.detect_persons_with_pose(frame)

                # 计算局部运动
                local_motion = self.compute_local_motion(persons, frame)

                # 计算帧间运动作为补充
                frame_motion = self.compute_frame_motion(frame)

                # 综合运动指标（侧重局部运动）
                combined_motion = local_motion * 0.7 + frame_motion * 0.3

                self.motion_log.append(combined_motion)
                self.local_motion_log.append(local_motion)

                # 输出
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    avg_motion = np.mean(self.motion_log) if self.motion_log else 0.0
                    avg_local = np.mean(self.local_motion_log) if self.local_motion_log else 0.0

                    level, color = self.get_motion_level(combined_motion)

                    print(f"局部运动: {local_motion:.4f} | 综合: {combined_motion:.4f} | "
                          f"平均: {avg_motion:.4f} | {color}{level}\033[0m | "
                          f"人数: {len(persons)}")

                    last_print_time = current_time

                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n监测结束")
        finally:
            self.is_running = False
            if self.motion_log:
                stats = {
                    'max': max(self.motion_log),
                    'avg': np.mean(self.motion_log),
                    'min': min(self.motion_log)
                }
                print(f"局部运动统计 - 最大: {stats['max']:.4f} | "
                      f"平均: {stats['avg']:.4f} | 最小: {stats['min']:.4f}")

    def stop_monitor(self):
        """停止监测"""
        self.is_running = False


# 独立运行
if __name__ == "__main__":
    monitor = LocalMotionMonitor('yolo11s.pt', conf_threshold=0.3)
    monitor.run_local_motion_monitor()
