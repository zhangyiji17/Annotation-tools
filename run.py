#!/usr/bin/env python3
import subprocess
import sys
import os
import torch
from pathlib import Path


def install_dependencies():
    """安装所需依赖"""
    print("正在安装依赖...")

    dependencies = [
        "gradio>=4.0.0",
        "requests>=2.28.0",
        "pymupdf>=1.23.0",
        "pandas>=2.0.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0"
    ]

    for dep in dependencies:
        print(f"安装 {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])

    print("✅ 所有依赖安装完成！")


def download_yolo_model():
    """下载YOLO模型"""
    print("正在下载YOLO模型...")

    # 创建模型目录
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    try:
        # 使用ultralytics自动下载模型
        from ultralytics import YOLO

        print("下载yolo11n.pt模型...")
        model = YOLO('yolo11n.pt')
        model_path = model_dir / "yolo11n.pt"

        # 保存模型到本地
        if hasattr(model, 'model'):
            torch.save(model.model.state_dict(), model_path)
        print(f"✅ 模型下载完成: {model_path}")

    except Exception as e:
        print(f"❌ 模型下载失败: {e}")
        print("将使用ultralytics的自动下载功能")


def main():
    """主启动函数"""
    print("=" * 60)
    print("🤖 AI数据标注工具 v2.0 - 启动器")
    print("=" * 60)

    # 检查依赖
    try:
        import gradio
        import ultralytics
        import cv2
        print("✅ 核心依赖检查通过")
    except ImportError:
        print("❌ 缺少核心依赖，正在安装...")
        install_dependencies()

    # 创建必要目录
    from config import Config
    Config.create_dirs()
    print("✅ 目录结构已创建")

    # 启动应用
    print("\n" + "=" * 60)
    print("正在启动AI数据标注工具...")
    print("访问地址: http://localhost:7861")
    print("=" * 60)
    print("功能说明:")
    print("1. 问答对生成: 支持PDF文件和文本输入")
    print("2. 图像标注: 基于YOLO11n的自动目标检测")
    print("3. 多模态生成: 图片内容分析")
    print("按 Ctrl+C 停止应用")
    print("=" * 60 + "\n")

    try:
        import main
        import gradio as gr

        app = main.create_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            debug=False,
            show_error=True,
            theme=gr.themes.Soft(),
            favicon_path="favicon.ico" if Path("favicon.ico").exists() else None
        )
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("请检查依赖安装和配置文件")


if __name__ == "__main__":
    main()