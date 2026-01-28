import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
import os


class YOLODetector:
    """YOLO目标检测器（支持目标检测、图像分类、实例分割）"""

    def __init__(self, model_path: str = None, task_subtype: str = "目标检测"):
        """
        初始化YOLO检测器
        :param model_path: 模型文件路径（可选，默认使用预训练模型）
        :param task_subtype: 任务子类型 - "目标检测"、"图像分类"、"实例分割"
        """
        # 清理任务类型字符串
        self.task_subtype = self._clean_task_subtype(task_subtype)

        # COCO数据集类别名称（共80类）
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        self.model_path = model_path
        self.model = None
        self._load_model()

    def _clean_task_subtype(self, task_subtype: str) -> str:
        """清理任务类型字符串"""
        if not isinstance(task_subtype, str):
            return "目标检测"

        # 移除常见的表情符号
        emoji_map = {
            "🎯 ": "",
            "🖼️ ": "",
            "🖌️ ": "",
            "🎯": "",
            "🖼️": "",
            "🖌️": ""
        }

        for emoji, replacement in emoji_map.items():
            task_subtype = task_subtype.replace(emoji, replacement)

        task_subtype = task_subtype.strip()

        # 映射到标准任务类型
        if "目标检测" in task_subtype:
            return "目标检测"
        elif "图像分类" in task_subtype:
            return "图像分类"
        elif "实例分割" in task_subtype:
            return "实例分割"
        else:
            return task_subtype or "目标检测"

    def _load_model(self):
        """加载YOLO模型"""
        try:
            from ultralytics import YOLO

            print(f"✅ Ultralytics YOLO 库已导入，任务类型: {self.task_subtype}")

            # 检查模型路径是否存在
            if self.model_path and os.path.exists(self.model_path):
                model_filename = os.path.basename(self.model_path).lower()

                # 检查是否是YOLOv5模型
                if 'yolov5' in model_filename:
                    print(f"⚠️  检测到YOLOv5模型: {self.model_path}")
                    print("YOLOv5模型在ultralytics中可能需要特殊处理...")

                    try:
                        # 尝试正常加载
                        self.model = YOLO(self.model_path, verbose=False)
                        print(f"✅ 成功加载模型: {self.model_path}")
                    except Exception as e:
                        print(f"❌ 加载模型失败: {e}")
                        print("尝试使用备用方法加载...")

                        # 备用方法：设置环境变量避免自动下载
                        os.environ['YOLO_VERBOSE'] = 'False'
                        try:
                            self.model = YOLO(self.model_path, task='detect')
                            print(f"✅ 使用备用方法成功加载模型")
                        except Exception as e2:
                            print(f"❌ 备用方法也失败: {e2}")
                            raise Exception(f"无法加载YOLOv5模型: {e2}")
                else:
                    # 对于非YOLOv5模型，正常加载
                    print(f"📦 加载模型: {self.model_path}")
                    self.model = YOLO(self.model_path, verbose=False)
                    print(f"✅ 成功加载模型: {self.model_path}")
            else:
                # 根据任务类型加载预训练模型
                print(f"⚠️  模型文件不存在: {self.model_path}")
                print("尝试加载预训练模型...")

                if self.task_subtype == "目标检测":
                    try:
                        self.model = YOLO('yolo11n.pt', verbose=False)
                        print("✅ 加载预训练YOLO11n模型（目标检测）")
                    except Exception as e:
                        print(f"⚠️  加载yolo11n失败: {e}")
                        # 如果yolo11n不可用，尝试yolov8n
                        try:
                            self.model = YOLO('yolov8n.pt', verbose=False)
                            print("✅ 加载预训练YOLOv8n模型（目标检测）")
                        except Exception as e2:
                            print(f"❌ 加载预训练模型失败: {e2}")
                            raise Exception("无法加载任何预训练模型")
                elif self.task_subtype == "图像分类":
                    try:
                        self.model = YOLO('yolov8n-cls.pt', verbose=False)
                        print("✅ 加载预训练YOLOv8n-cls模型（图像分类）")
                    except Exception as e:
                        print(f"❌ 加载分类模型失败: {e}")
                        raise Exception("无法加载分类模型")
                elif self.task_subtype == "实例分割":
                    try:
                        self.model = YOLO('yolov8n-seg.pt', verbose=False)
                        print("✅ 加载预训练YOLOv8n-seg模型（实例分割）")
                    except Exception as e:
                        print(f"❌ 加载分割模型失败: {e}")
                        raise Exception("无法加载分割模型")
                else:
                    print(f"❌ 未知任务类型: {self.task_subtype}")
                    raise Exception(f"未知任务类型: {self.task_subtype}")

        except ImportError as e:
            print(f"❌ 未安装ultralytics: {e}")
            print("请运行: pip install ultralytics")
            raise
        except Exception as e:
            print(f"❌ 加载模型失败: {str(e)}")
            raise

    def detect(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        检测图片中的目标
        """
        if self.model is None:
            print("❌ 模型未加载")
            return []

        try:
            # 读取图片获取尺寸
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ 无法读取图片: {image_path}")
                return []

            img_height, img_width = img.shape[:2]

            # 执行检测
            results = self.model(image_path, conf=conf_threshold, verbose=False)

            if not results:
                return []

            detections = []

            # 解析结果
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()

                    for i, box in enumerate(boxes):
                        # 获取坐标和置信度
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])

                        if confidence >= conf_threshold:
                            # 转换为归一化坐标
                            center_x = ((x1 + x2) / 2) / img_width
                            center_y = ((y1 + y2) / 2) / img_height
                            width = (x2 - x1) / img_width
                            height = (y2 - y1) / img_height

                            # 确保坐标在[0,1]范围内
                            center_x = max(0, min(1, center_x))
                            center_y = max(0, min(1, center_y))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))

                            # 获取类别名称
                            if class_id < len(self.class_names):
                                class_name = self.class_names[class_id]
                            else:
                                class_name = f"class_{class_id}"

                            detections.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': float(confidence),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'yolo_bbox': [float(center_x), float(center_y), float(width), float(height)],
                                'image_width': img_width,
                                'image_height': img_height,
                                'task_type': 'detection'
                            })

            return detections

        except Exception as e:
            print(f"❌ 目标检测失败: {str(e)}")
            return []

    def classify(self, image_path: str, conf_threshold: float = 0.25, top_n: int = 5) -> List[Dict]:
        """图像分类"""
        # 简化实现：使用目标检测结果作为分类结果
        print(f"⚠️  图像分类功能当前使用目标检测模型模拟实现")
        detections = self.detect(image_path, conf_threshold)

        if not detections:
            return []

        # 统计类别
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # 转换为分类结果
        classifications = []
        total = len(detections)

        for i, (class_name, count) in enumerate(class_counts.items()):
            confidence = count / total if total > 0 else 0

            classifications.append({
                'class_id': i,
                'class_name': class_name,
                'confidence': float(confidence),
                'rank': i + 1,
                'image_width': detections[0]['image_width'],
                'image_height': detections[0]['image_height'],
                'task_type': 'classification',
                'top_n': len(class_counts)
            })

        return classifications

    def segment(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """实例分割"""
        # 简化实现：使用目标检测结果作为分割结果
        print(f"⚠️  实例分割功能当前使用目标检测模型模拟实现")
        detections = self.detect(image_path, conf_threshold)

        if not detections:
            return []

        # 转换为分割结果格式
        segmentations = []

        for det in detections:
            segmentations.append({
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'yolo_bbox': det['yolo_bbox'],
                'mask': [],  # 空掩码
                'image_width': det['image_width'],
                'image_height': det['image_height'],
                'task_type': 'segmentation'
            })

        return segmentations

    def visualize(self, image_path: str, detections: List[Dict], output_path: str = None) -> np.ndarray:
        """
        可视化检测结果
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        img_draw = img.copy()

        # 定义颜色
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (255, 165, 0)]

        for i, det in enumerate(detections):
            color = colors[i % len(colors)]

            if 'bbox' in det:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)

                # 绘制标签
                label = f"{det.get('class_name', 'unknown')}: {det.get('confidence', 0):.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                cv2.rectangle(img_draw,
                              (x1, y1 - label_height - 10),
                              (x1 + label_width, y1),
                              color, -1)

                cv2.putText(img_draw, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

        if output_path:
            cv2.imwrite(output_path, img_draw)

        return img_draw

    def save_yolo_format(self, image_name: str, detections: List[Dict], output_path: str):
        """保存为YOLO格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for det in detections:
                if 'yolo_bbox' in det:
                    class_id = det.get('class_id', 0)
                    center_x, center_y, width, height = det['yolo_bbox']
                    line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                    f.write(line)

    def save_coco_format(self, image_name: str, detections: List[Dict], output_path: str):
        """保存为COCO格式"""
        annotations = []

        for i, det in enumerate(detections):
            if 'bbox' in det:
                x1, y1, x2, y2 = det['bbox']
                width = x2 - x1
                height = y2 - y1

                annotation = {
                    'id': i + 1,
                    'image_id': image_name,
                    'category_id': det.get('class_id', 0),
                    'category_name': det.get('class_name', 'unknown'),
                    'bbox': [float(x1), float(y1), float(width), float(height)],
                    'area': float(width * height),
                    'confidence': det.get('confidence', 0),
                    'segmentation': [],
                    'iscrowd': 0
                }
                annotations.append(annotation)

        result = {
            'image_name': image_name,
            'image_size': {
                'width': detections[0]['image_width'] if detections else 0,
                'height': detections[0]['image_height'] if detections else 0
            },
            'detections': annotations
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def save_classification_format(self, image_name: str, classifications: List[Dict], output_path: str):
        """保存分类结果格式"""
        result = {
            'image_name': image_name,
            'image_size': {
                'width': classifications[0]['image_width'] if classifications else 0,
                'height': classifications[0]['image_height'] if classifications else 0
            },
            'task_type': 'classification',
            'top_n': len(classifications),
            'classifications': classifications
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def save_segmentation_format(self, image_name: str, segmentations: List[Dict], output_path: str):
        """保存分割结果格式"""
        result = {
            'image_name': image_name,
            'image_size': {
                'width': segmentations[0]['image_width'] if segmentations else 0,
                'height': segmentations[0]['image_height'] if segmentations else 0
            },
            'task_type': 'segmentation',
            'segmentations': segmentations
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)