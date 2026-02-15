import numpy as np
import time
from collections import deque
from mss import mss
from ultralytics import YOLO


class PersonMotionMonitor:
    def __init__(self, model_path='yolo11s.pt', conf_threshold=0.5):
        print("初始化人员运动监测器...")

        try:
            self.model = YOLO(model_path)
            print(f"模型加载成功: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise

        self.conf_threshold = conf_threshold
        self.prev_person_centers = None
        self.motion_log = deque(maxlen=20)
        self.sct = mss()
        self.monitor = self.sct.monitors[1]
        self.frame_count = 0
        self.start_time = time.time()

        # 获取person类别的ID
        self.person_class_id = None
        for class_id, class_name in self.model.names.items():
            if class_name == 'person':
                self.person_class_id = class_id
                break

        if self.person_class_id is None:
            print("警告: 未找到'person'类别，将使用所有检测到的人员")

    def get_screen(self):
        """安全地获取屏幕图像"""
        try:
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)
            # 确保是3通道图像
            return img[:, :, :3] if img.shape[2] == 4 else img
        except Exception as e:
            print(f"屏幕捕获错误: {e}")
            return None

    def detect_persons(self, image):
        """只检测人员"""
        if image is None:
            return []

        try:
            results = self.model(image, verbose=False, conf=self.conf_threshold)
            persons = []

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()

                for box, cls, conf in zip(boxes, classes, confidences):
                    # 只检测person类别
                    if int(cls) == self.person_class_id or self.model.names[int(cls)] == 'person':
                        x1, y1, x2, y2 = box
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        persons.append({
                            'center': (cx, cy),
                            'box': box,
                            'confidence': conf
                        })

            return persons

        except Exception as e:
            print(f"人员检测错误: {e}")
            return []

    def compute_person_motion(self, current_persons):
        """计算人员运动"""
        if not current_persons:
            self.prev_person_centers = None
            return 0.0

        if self.prev_person_centers is None:
            self.prev_person_centers = [person['center'] for person in current_persons]
            return 0.0

        motion_sum = 0.0
        count = 0

        for curr_person in current_persons:
            min_dist = float('inf')
            curr_center = curr_person['center']

            for prev_center in self.prev_person_centers:
                dist = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)
                if dist < min_dist:
                    min_dist = dist

            # 归一化运动距离
            if min_dist < float('inf'):
                screen_width = self.monitor['width']
                screen_height = self.monitor['height']
                max_possible_distance = np.sqrt(screen_width ** 2 + screen_height ** 2)
                normalized_motion = min(min_dist / (max_possible_distance * 0.1), 1.0)
                motion_sum += normalized_motion
                count += 1

        self.prev_person_centers = [person['center'] for person in current_persons]

        return motion_sum / count if count > 0 else 0.0

    def get_motion_level(self, motion_rate):
        """获取运动等级"""
        if motion_rate < 0.02:
            return "静止", "\033[90m"  # 灰色
        elif motion_rate < 0.08:
            return "微动", "\033[92m"  # 绿色
        elif motion_rate < 0.2:
            return "缓慢", "\033[93m"  # 黄色
        elif motion_rate < 0.4:
            return "中等", "\033[33m"  # 橙色
        else:
            return "剧烈", "\033[91m"  # 红色

    def run_monitor(self):
        """主监控循环"""
        print("开始人员运动监测")
        print("只检测 'person' 类别")
        print("按 Ctrl+C 停止")
        print("=" * 60)

        last_print_time = time.time()
        print_interval = 1.0  # 每秒打印一次

        try:
            while True:
                # 获取和处理图像
                image = self.get_screen()
                if image is None:
                    time.sleep(0.1)
                    continue

                # 检测人员
                persons = self.detect_persons(image)
                person_count = len(persons)

                # 计算人员运动
                motion = self.compute_person_motion(persons)
                self.motion_log.append(motion)

                # 定期打印状态
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    avg_motion = np.mean(self.motion_log) if self.motion_log else 0.0

                    # 运动等级
                    level, color = self.get_motion_level(motion)

                    print(f"人员运动: {motion:.4f} | 平均: {avg_motion:.4f} | {color}{level}\033[0m | "
                          f"人数: {person_count}")

                    last_print_time = current_time

                time.sleep(0.05)  # 控制检测频率

        except KeyboardInterrupt:
            print("\n监测结束")
            if self.motion_log:
                final_avg = np.mean(self.motion_log)
                print(f"最终平均人员运动速率: {final_avg:.4f}")


# 运行
if __name__ == "__main__":
    monitor = PersonMotionMonitor('yolo11s.pt', conf_threshold=0.5)
    monitor.run_monitor()
