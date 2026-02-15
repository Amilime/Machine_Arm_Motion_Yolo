# -*- coding: utf-8 -*-
# @Author  : Amilime
# @Time    : 10/31/2025 11:02 PM
# @File    : screen_catch.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
# @Author  : Amilime
# @Time    : 10/31/2025 11:02 PM
# @File    : screen_catch.py
# @Software: PyCharm

import cv2
import numpy as np
from collections import deque
import time
import pyautogui
# from PIL import Image


class MotionIntensityDetector:
    def __init__(self, window_size=30, roi=None, intensity_threshold=10):
        """
        初始化运动强度检测器

        参数:
            window_size: 滑动窗口大小，用于计算平均运动强度
            roi: 感兴趣区域 (x, y, width, height)，如果为None则检测整个屏幕
            intensity_threshold: 运动强度阈值，用于过滤噪声
        """
        self.window_size = window_size
        self.roi = roi
        self.intensity_threshold = intensity_threshold

        # 存储最近的运动强度值
        self.motion_history = deque(maxlen=window_size)

        # 前一帧图像
        self.prev_frame = None

        # 运动权重参数
        self.base_weight = 0.0
        self.max_weight = 10.0

    def preprocess_frame(self, frame):
        """预处理帧"""
        if self.roi is not None:
            x, y, w, h = self.roi
            frame = frame[y:y + h, x:x + w]

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 高斯模糊以减少噪声
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        return blurred

    def calculate_motion_intensity(self, current_frame):
        """计算当前帧与前一帧之间的运动强度"""
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return 0.0

        # 计算帧间差异
        frame_diff = cv2.absdiff(self.prev_frame, current_frame)

        # 二值化处理
        _, thresh = cv2.threshold(frame_diff, self.intensity_threshold, 255, cv2.THRESH_BINARY)

        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 计算运动强度（非零像素的比例）
        motion_intensity = np.sum(thresh > 0) / (thresh.shape[0] * thresh.shape[1])

        # 更新前一帧
        self.prev_frame = current_frame

        return motion_intensity

    def calculate_weight(self, motion_intensity):
        """根据运动强度计算权重"""
        # 将运动强度映射到权重范围
        weight = motion_intensity * self.max_weight

        # 应用非线性映射，但避免过度放大
        # 使用更平缓的曲线
        if weight > 0:
            weight = np.sqrt(weight) * 3.0  # 降低放大系数

        # 添加历史平滑
        if self.motion_history:
            avg_intensity = np.mean(list(self.motion_history))
            # 当前值和历史平均值加权
            weight = weight * 0.7 + (avg_intensity * self.max_weight) * 0.3

        # 限制权重范围
        weight = np.clip(weight, self.base_weight, self.max_weight)

        return weight

    def process_frame(self, frame):
        """处理单帧图像并返回运动权重"""
        # 预处理
        processed_frame = self.preprocess_frame(frame)

        # 计算运动强度
        motion_intensity = self.calculate_motion_intensity(processed_frame)

        # 更新历史记录
        self.motion_history.append(motion_intensity)

        # 计算当前权重
        current_weight = self.calculate_weight(motion_intensity)

        # 计算平均权重（平滑处理）
        avg_weight = np.mean(list(self.motion_history)) * self.max_weight if self.motion_history else 0

        return {
            'current_weight': current_weight,
            'average_weight': avg_weight,
            'motion_intensity': motion_intensity,
            'processed_frame': processed_frame
        }


class ScreenCapture:
    """真正的屏幕捕获类"""

    def __init__(self, screen_region=None, reduce_scale=0.5):
        """
        初始化屏幕捕获

        参数:
            screen_region: 捕获区域 (x, y, width, height)，None为全屏
            reduce_scale: 缩放比例，降低分辨率以提高处理速度
        """
        self.screen_region = screen_region
        self.reduce_scale = reduce_scale

    def capture_screen(self):
        """捕获屏幕画面"""
        try:
            # 捕获屏幕
            if self.screen_region:
                screenshot = pyautogui.screenshot(region=self.screen_region)
            else:
                screenshot = pyautogui.screenshot()

            # 转换为OpenCV格式
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # 降低分辨率以提高处理速度
            if self.reduce_scale != 1.0:
                new_width = int(frame.shape[1] * self.reduce_scale)
                new_height = int(frame.shape[0] * self.reduce_scale)
                frame = cv2.resize(frame, (new_width, new_height))

            return frame

        except Exception as e:
            print(f"屏幕捕获错误: {e}")
            return None


class GameMotionAnalyzer:
    """游戏运动分析器 - 专门针对游戏场景优化"""

    def __init__(self, game_region=None, sensitivity=1.0):
        """
        初始化游戏运动分析器

        参数:
            game_region: 游戏区域 (x, y, width, height)
            sensitivity: 敏感度 (0.1 - 2.0)
        """
        self.game_region = game_region
        self.sensitivity = sensitivity

        # 初始化运动检测器
        self.detector = MotionIntensityDetector(
            window_size=20,
            intensity_threshold=25,
            roi=game_region
        )

        # 屏幕捕获
        self.screen_capture = ScreenCapture(
            screen_region=game_region,
            reduce_scale=0.3  # 降低分辨率以提高性能
        )

        # 运动状态
        self.motion_levels = {
            'low': (0, 2),
            'medium': (2, 5),
            'high': (5, 8),
            'extreme': (8, 10)
        }

    def get_motion_level(self, weight):
        """获取运动级别"""
        for level, (min_val, max_val) in self.motion_levels.items():
            if min_val <= weight < max_val:
                return level
        return 'extreme'

    def analyze_game_motion(self, duration=60):
        """分析游戏运动"""
        print(f"开始分析游戏运动，持续时间: {duration}秒")
        print("按 'q' 键提前退出")

        start_time = time.time()
        motion_data = []

        while time.time() - start_time < duration:
            # 捕获屏幕
            frame = self.screen_capture.capture_screen()
            if frame is None:
                continue

            # 处理帧
            result = self.detector.process_frame(frame)

            # 获取结果
            current_weight = result['current_weight']
            motion_level = self.get_motion_level(current_weight)

            # 存储数据
            motion_data.append({
                'timestamp': time.time() - start_time,
                'weight': current_weight,
                'level': motion_level,
                'intensity': result['motion_intensity']
            })

            # 显示实时信息
            self.display_realtime_info(frame, result, motion_level)

            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 分析结果
        self.generate_report(motion_data)
        cv2.destroyAllWindows()

        return motion_data

    def display_realtime_info(self, frame, result, motion_level):
        """显示实时信息"""
        current_weight = result['current_weight']
        avg_weight = result['average_weight']
        motion_intensity = result['motion_intensity']

        # 根据运动级别设置颜色
        color_map = {
            'low': (0, 255, 0),  # 绿色
            'medium': (0, 255, 255),  # 黄色
            'high': (0, 165, 255),  # 橙色
            'extreme': (0, 0, 255)  # 红色
        }
        color = color_map.get(motion_level, (255, 255, 255))

        # 在画面上显示信息
        cv2.putText(frame, f"Motion Level: {motion_level.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Current Weight: {current_weight:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Average Weight: {avg_weight:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Intensity: {motion_intensity:.4f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 创建运动强度条
        intensity_bar = self.create_intensity_bar(current_weight, motion_level)

        # 显示画面
        cv2.imshow('Game Motion Analysis', frame)
        cv2.imshow('Motion Intensity', intensity_bar)

    def create_intensity_bar(self, weight, level):
        """创建运动强度可视化条"""
        bar_width = 400
        bar_height = 60
        intensity_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        # 填充颜色
        fill_width = int(weight / 10.0 * bar_width)
        color_map = {
            'low': (0, 255, 0),  # 绿色
            'medium': (0, 255, 255),  # 黄色
            'high': (0, 165, 255),  # 橙色
            'extreme': (0, 0, 255)  # 红色
        }
        color = color_map.get(level, (255, 255, 255))

        cv2.rectangle(intensity_bar, (0, 0), (fill_width, bar_height), color, -1)
        cv2.rectangle(intensity_bar, (0, 0), (bar_width, bar_height), (255, 255, 255), 1)

        # 添加刻度
        for i in range(0, 11, 2):
            x_pos = int(i / 10.0 * bar_width)
            cv2.line(intensity_bar, (x_pos, bar_height - 10), (x_pos, bar_height), (255, 255, 255), 1)
            cv2.putText(intensity_bar, str(i), (x_pos - 5, bar_height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        cv2.putText(intensity_bar, f"Motion Intensity: {level.upper()}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return intensity_bar

    def generate_report(self, motion_data):
        """生成分析报告"""
        if not motion_data:
            print("没有收集到数据")
            return

        weights = [data['weight'] for data in motion_data]
        max_weight = max(weights)
        avg_weight = np.mean(weights)

        print("\n" + "=" * 50)
        print("游戏运动分析报告")
        print("=" * 50)
        print(f"分析时长: {motion_data[-1]['timestamp']:.1f} 秒")
        print(f"最大运动权重: {max_weight:.2f}")
        print(f"平均运动权重: {avg_weight:.2f}")
        print(f"总体运动级别: {self.get_motion_level(avg_weight).upper()}")


# 使用示例
def main():
    # 方法1: 全屏检测
    print("方法1: 全屏运动检测")
    analyzer = GameMotionAnalyzer()
    analyzer.analyze_game_motion(duration=30)  # 分析30秒

    # 方法2: 指定游戏区域检测（更精确）
    # 你需要先确定游戏窗口的位置和大小
    # game_region = (100, 100, 800, 600)  # (x, y, width, height)
    # analyzer = GameMotionAnalyzer(game_region=game_region)
    # analyzer.analyze_game_motion(duration=60)


def quick_start():
    """快速开始 - 最简单的使用方法"""
    print("快速开始: 屏幕运动检测")
    print("将检测整个屏幕的运动变化")
    print("按 'q' 键退出")

    # 初始化检测器
    detector = MotionIntensityDetector(
        window_size=20,
        intensity_threshold=20
    )

    # 初始化屏幕捕获
    screen_capture = ScreenCapture(reduce_scale=0.5)  # 降低分辨率提高速度

    while True:
        # 捕获屏幕
        frame = screen_capture.capture_screen()
        if frame is None:
            continue

        # 处理帧
        result = detector.process_frame(frame)

        # 显示结果
        current_weight = result['current_weight']
        print(f"\r当前运动权重: {current_weight:.2f} | 运动强度: {result['motion_intensity']:.4f}", end="")

        # 退出条件
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n检测结束")


if __name__ == "__main__":
    # 安装依赖: pip install pyautogui opencv-python numpy pillow

    # 选择运行模式:

    # 模式1: 快速开始（简单输出）
    quick_start()

    # 模式2: 完整分析（带可视化界面）
    # main()
