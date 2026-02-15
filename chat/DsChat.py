import requests
import json
import time


def monitor_tavern_requests():
    """通过创建测试会话来发现API端点"""

    session = requests.Session()

    # 先访问主页获取cookies
    print("获取初始会话...")
    response = session.get("http://127.0.0.1:8000/")
    print(f"主页状态: {response.status_code}")

    # 尝试发现API端点
    discovery_endpoints = [
        "/api",
        "/api/",
        "/api/chat",
        "/api/v1",
        "/api/v1/chat",
        "/rest/api",
        "/plugins",
        "/api/plugins"
    ]

    for endpoint in discovery_endpoints:
        url = f"http://127.0.0.1:8000{endpoint}"
        print(f"探索: {url}")
        try:
            response = session.get(url, timeout=5)
            print(f"  状态: {response.status_code}")
            if response.status_code == 200:
                print(f"  内容类型: {response.headers.get('content-type')}")
        except Exception as e:
            print(f"  错误: {e}")


monitor_tavern_requests()
