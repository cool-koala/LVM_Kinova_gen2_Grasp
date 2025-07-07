#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python Script Monitor for ROS Integration
监视共享目录中生成的Python脚本，并自动执行

This script monitors a shared directory for generated Python files and executes them
to publish ROS topics for robot control integration.

监视共享目录中生成的Python文件并执行它们以发布ROS主题进行机器人控制集成

Author: Auto-generated for LVM_Kinova_gen2_Grasp project
"""

import os
import time
import subprocess
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class PythonFileHandler(FileSystemEventHandler):
    """监控Python文件变化的处理器"""
    
    def __init__(self, target_files=None):
        super().__init__()
        self.target_files = target_files or ['generated.py', 'genzerated.py']
        self.execution_lock = threading.Lock()
        print(f"开始监控文件: {self.target_files}")
    
    def on_created(self, event):
        """文件创建时的回调"""
        if not event.is_directory:
            self.handle_file_event(event.src_path, "created")
    
    def on_modified(self, event):
        """文件修改时的回调"""
        if not event.is_directory:
            self.handle_file_event(event.src_path, "modified")
    
    def handle_file_event(self, file_path, event_type):
        """处理文件事件"""
        filename = os.path.basename(file_path)
        
        # 检查是否为目标文件
        if filename in self.target_files:
            print(f"检测到文件 {event_type}: {file_path}")
            
            # 等待文件写入完成
            time.sleep(0.5)
            
            # 执行Python文件
            self.execute_python_file(file_path)
    
    def execute_python_file(self, file_path):
        """执行Python文件"""
        with self.execution_lock:
            try:
                print(f"正在执行: {file_path}")
                
                # 检查文件是否可读
                if not os.path.isfile(file_path):
                    print(f"文件不存在: {file_path}")
                    return
                
                # 执行Python文件
                result = subprocess.run([
                    'python', file_path
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"成功执行 {file_path}")
                    if result.stdout:
                        print(f"输出: {result.stdout}")
                else:
                    print(f"执行失败 {file_path}, 返回码: {result.returncode}")
                    if result.stderr:
                        print(f"错误: {result.stderr}")
                
                # 执行完成后删除文件（可选）
                try:
                    os.remove(file_path)
                    print(f"已删除执行完成的文件: {file_path}")
                except OSError as e:
                    print(f"删除文件失败: {e}")
                    
            except subprocess.TimeoutExpired:
                print(f"执行超时: {file_path}")
            except Exception as e:
                print(f"执行出错: {e}")

def monitor_directory(watch_directory, target_files=None):
    """监控指定目录"""
    if not os.path.exists(watch_directory):
        print(f"监控目录不存在: {watch_directory}")
        print("请确保共享目录已正确挂载")
        return
    
    print(f"开始监控目录: {watch_directory}")
    
    # 创建事件处理器
    event_handler = PythonFileHandler(target_files)
    
    # 创建观察者
    observer = Observer()
    observer.schedule(event_handler, watch_directory, recursive=False)
    
    # 启动监控
    observer.start()
    print("Python脚本监控器已启动...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止监控...")
        observer.stop()
    
    observer.join()
    print("监控已停止")

def main():
    """主函数"""
    # 默认监控目录 - 用户需要根据实际情况修改IP地址
    # Default monitoring directory - users need to modify IP address based on actual setup
    default_directory = "/home/guoxy/Camera_capture"  # 本地目录示例
    
    # 可选的网络共享目录（需要先挂载）
    # Optional network shared directory (needs to be mounted first)
    # network_directory = "//YOUR_IP_HERE/Camera_capture"
    
    # 目标文件列表
    target_files = ['generated.py', 'genzerated.py']
    
    print("=" * 50)
    print("Python Script Monitor for ROS Integration")
    print("Python脚本监控器 - ROS集成")
    print("=" * 50)
    print(f"监控目录: {default_directory}")
    print(f"目标文件: {target_files}")
    print("=" * 50)
    
    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(default_directory):
        try:
            os.makedirs(default_directory)
            print(f"已创建监控目录: {default_directory}")
        except OSError as e:
            print(f"无法创建目录: {e}")
            return
    
    # 开始监控
    monitor_directory(default_directory, target_files)

if __name__ == "__main__":
    main()