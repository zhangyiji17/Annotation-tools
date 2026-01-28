import os
import torch
from pathlib import Path

# API配置
API_CONFIG = {
    "DeepSeek": {
        "base_url": "http://120.236.144.102:3001/v1",
        "default_model": "oneapi-DeepSeek-V3",
        "supports_vision": False
    },
    "文心一言": {
        "base_url": "https://qianfan.baidubce.com",
        "default_model": "ERNIE-Bot",
        "supports_vision": True
    },
    "ChatGPT4.0": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4",
        "supports_vision": True
    }
}

# 预设提示词
PRESETS = {
    "能碳知识查询": {
        "system_prompt": "你是一个专业的工程机械行业能碳领域科学工作者，请根据知识文档生成简洁、专业、工整的知识领域问答对",
        "task_prompt": "请生成概念性、工整的知识问答对，避免包含页码、编号、引用或特定公司信息"
    },
    "设备维护检查": {
        "system_prompt": "你是一个专业的设备管理员，请根据技术文档生成简洁的设备维护问答对",
        "task_prompt": "请生成概念性的技术问答对，避免包含页码、编号、引用或特定公司信息"
    },
    "故障诊断分析": {
        "system_prompt": "你是一个经验丰富的设备维修师",
        "task_prompt": "请生成通用的故障诊断问答对，不要包含特定案例或公司信息"
    },
    "技术参数查询": {
        "system_prompt": "你是技术参数专家",
        "task_prompt": "请生成通用的技术参数问答，避免包含特定产品或公司信息"
    },
    "安全规程生成": {
        "system_prompt": "你是安全生产专家，需制定安全操作规程",
        "task_prompt": "请生成通用的工业安全问答对，不要包含特定公司规程"
    },
    "操作手册问答": {
        "system_prompt": "你是操作手册解析专家",
        "task_prompt": "请生成通用的设备操作步骤问答对，避免包含特定型号或公司信息"
    },
    "运营指标问答": {
        "system_prompt": "你是企业运营指标解析专家",
        "task_prompt": "请生成通用的运营指标问答对，避免包含页码、编号、引用或特定公司信息"
    }
}

# YOLO模型配置
YOLO_CONFIG = {
    "model_path": "models/yolo11n.pt",  # 修改为yolo11n.pt
    "conf_threshold": 0.25,  # 置信度阈值
    "iou_threshold": 0.45,  # IOU阈值
    "device": "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备
}


class Config:
    UPLOAD_DIR = "uploads"
    OUTPUT_DIR = "outputs"
    ANNOTATION_DIR = "outputs/annotations"
    VISUALIZATION_DIR = "outputs/visualizations"
    MODEL_DIR = "models"

    # YOLO模型路径 - 修改为yolo11n.pt
    YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolo11n.pt")

    @staticmethod
    def create_dirs():
        """创建必要的目录"""
        dirs = [
            Config.UPLOAD_DIR,
            Config.OUTPUT_DIR,
            Config.ANNOTATION_DIR,
            Config.VISUALIZATION_DIR,
            Config.MODEL_DIR,
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"✅ 确保目录存在: {dir_path}")

    @staticmethod
    def get_default_model_path():
        """获取默认模型路径"""
        return Config.YOLO_MODEL_PATH

    @staticmethod
    def is_default_model_exists():
        """检查默认模型是否存在"""
        return os.path.exists(Config.YOLO_MODEL_PATH)