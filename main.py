import gradio as gr
import os
import json
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Generator, Tuple, Any, Optional
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw
import base64
import io
import shutil
import uuid

from config import Config, PRESETS, API_CONFIG
from api_handler import APIHandler
from pdf_processor import PDFProcessor
from yolo_detector import YOLODetector

# 创建目录
Config.create_dirs()

# 初始化YOLO检测器（懒加载）
yolo_detector = None
current_yolo_model_path = None
current_task_subtype = "目标检测"

# 全局变量存储当前处理结果
current_results = []
current_task_type = ""
current_file_paths = []
current_visualizations = []
current_detections = {}
current_original_images = {}
current_editing_mode = False
current_edit_image_index = -1
current_edit_annotations = []


def get_yolo_detector(model_path=None, task_subtype="目标检测"):
    """获取或初始化YOLO检测器"""
    global yolo_detector, current_yolo_model_path, current_task_subtype

    # 清理任务类型字符串
    if isinstance(task_subtype, str):
        task_subtype = clean_task_subtype(task_subtype)

    # 确定模型路径
    if model_path is not None:
        final_model_path = model_path
    elif current_yolo_model_path is not None:
        final_model_path = current_yolo_model_path
    else:
        final_model_path = Config.YOLO_MODEL_PATH

    print(f"🔄 初始化YOLO检测器: 模型={final_model_path}, 任务类型={task_subtype}")

    try:
        yolo_detector = YOLODetector(model_path=final_model_path, task_subtype=task_subtype)
        current_yolo_model_path = final_model_path
        current_task_subtype = task_subtype
        return yolo_detector
    except Exception as e:
        print(f"❌ 初始化YOLO检测器失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def clean_task_subtype(task_subtype: str) -> str:
    """清理任务类型字符串，移除表情符号"""
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

    # 确保是有效的任务类型
    valid_types = ["目标检测", "图像分类", "实例分割"]
    if task_subtype.strip() in valid_types:
        return task_subtype.strip()
    else:
        return "目标检测"


def test_api_connection(api_key: str, model_type: str):
    """测试API连接"""
    if not api_key.strip():
        return "❌ 请输入API密钥"

    try:
        handler = APIHandler(model_type, api_key.strip())
        if handler.test_connection():
            return f"✅ {model_type} 连接成功"
        else:
            return f"❌ {model_type} 连接失败，请检查API密钥"
    except Exception as e:
        return f"❌ 连接测试失败: {str(e)}"


def update_text_input_group(task_type_value):
    """根据任务类型更新文本输入组的显示"""
    if task_type_value == "问答对生成":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def update_file_upload(task_type_value):
    """根据任务类型更新文件上传组件"""
    if task_type_value == "问答对生成":
        return gr.update(file_types=[".pdf"], visible=True)
    elif task_type_value == "图像任务":
        return gr.update(file_types=[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"], visible=True)
    else:  # 多模态生成
        return gr.update(file_types=[".jpg", ".jpeg", ".png", ".bmp", ".gif"], visible=True)


def update_ui_components(task_type_value):
    """根据任务类型更新所有UI组件"""
    updates = []

    if task_type_value == "问答对生成":
        updates = [
            gr.update(visible=True),  # model_type
            gr.update(visible=True),  # api_key
            gr.update(visible=True),  # test_btn
            gr.update(visible=True),  # test_output
            gr.update(visible=False),  # yolo_config_group
            gr.update(visible=True),  # preset_type
            gr.update(visible=True),  # concurrency
            gr.update(visible=False),  # visualization_group
            gr.update(visible=False),  # image_navigation_group
            gr.update(value="问答对生成"),  # 更新任务类型状态
            gr.update(visible=False),  # 隐藏vision_model
            gr.update(visible=True),  # 显示自定义提示词组
            gr.update(visible=True),  # 显示文本输入组
            gr.update(visible=False),  # 隐藏编辑按钮
            gr.update(visible=False)  # 隐藏编辑控制组
        ]
    elif task_type_value == "图像任务":
        updates = [
            gr.update(visible=False),  # model_type
            gr.update(visible=False),  # api_key
            gr.update(visible=False),  # test_btn
            gr.update(visible=False),  # test_output
            gr.update(visible=True),  # yolo_config_group
            gr.update(visible=False),  # preset_type
            gr.update(visible=False),  # concurrency
            gr.update(visible=True),  # visualization_group
            gr.update(visible=True),  # image_navigation_group
            gr.update(value="图像任务"),  # 更新任务类型状态
            gr.update(visible=False),  # 隐藏vision_model
            gr.update(visible=False),  # 隐藏自定义提示词组
            gr.update(visible=False),  # 隐藏文本输入组
            gr.update(visible=True),  # 显示编辑按钮
            gr.update(visible=False)  # 隐藏编辑控制组（初始状态）
        ]
    else:  # 多模态生成
        updates = [
            gr.update(visible=True),  # model_type
            gr.update(visible=True),  # api_key
            gr.update(visible=True),  # test_btn
            gr.update(visible=True),  # test_output
            gr.update(visible=False),  # yolo_config_group
            gr.update(visible=False),  # preset_type
            gr.update(visible=True),  # concurrency
            gr.update(visible=False),  # visualization_group
            gr.update(visible=False),  # image_navigation_group
            gr.update(value="多模态生成"),  # 更新任务类型状态
            gr.update(visible=True),  # 显示vision_model
            gr.update(visible=False),  # 隐藏自定义提示词组
            gr.update(visible=False),  # 隐藏文本输入组
            gr.update(visible=False),  # 隐藏编辑按钮
            gr.update(visible=False)  # 隐藏编辑控制组
        ]

    return updates


def handle_local_model_upload(file):
    """处理本地模型上传"""
    global current_yolo_model_path

    if file is None:
        return (
            gr.update(value=None),  # 清除文件输入
            gr.update(value="请上传本地.pt模型文件"),  # 模型状态
            gr.update(value="本地模型")  # 更新模型选择按钮
        )

    try:
        # 获取上传的文件路径
        if hasattr(file, 'name'):
            model_path = file.name
        else:
            model_path = str(file)

        print(f"📁 上传模型文件: {model_path}")

        # 验证文件扩展名
        if not model_path.lower().endswith('.pt'):
            return (
                gr.update(value=None),
                gr.update(value="❌ 请选择.pt格式的模型文件"),
                gr.update(value="yolo11n.pt")  # 恢复到默认模型
            )

        # 检查文件是否存在
        if not os.path.exists(model_path):
            return (
                gr.update(value=None),
                gr.update(value=f"❌ 模型文件不存在: {model_path}"),
                gr.update(value="yolo11n.pt")
            )

        # 获取文件信息
        filename = os.path.basename(model_path)
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB

        # 如果是yolov5模型，警告用户可能存在问题
        warning_msg = ""
        if 'yolov5' in filename.lower():
            warning_msg = f"⚠️ 注意：YOLOv5模型在加载时可能会尝试加载带'u'后缀的版本，如果失败请尝试使用YOLOv8或YOLOv11模型"

        # 检查models目录是否已存在同名文件
        saved_model_path = os.path.join(Config.MODEL_DIR, filename)

        if os.path.exists(saved_model_path):
            print(f"🔄 发现同名模型文件，将覆盖替换: {saved_model_path}")
            # # 备份原有文件（可选）
            # backup_path = f"{saved_model_path}.backup_{int(time.time())}"
            # try:
            #     shutil.copy2(saved_model_path, backup_path)
            #     print(f"📦 已备份原文件到: {backup_path}")
            # except Exception as backup_error:
            #     print(f"⚠️  备份原文件失败: {backup_error}")

            # 覆盖替换
            shutil.copy2(model_path, saved_model_path)
            print(f"✅ 模型已覆盖: {saved_model_path}")
        else:
            # 直接复制新文件
            shutil.copy2(model_path, saved_model_path)
            print(f"✅ 模型已保存到: {saved_model_path}")

        # 更新全局变量
        current_yolo_model_path = saved_model_path

        status_msg = f"✅ 已选择本地模型: {filename} ({file_size:.1f}MB)"
        print(status_msg)

        if warning_msg:
            status_msg += f"\n{warning_msg}"

        return (
            gr.update(value=None),  # 清除文件输入
            gr.update(value=status_msg),  # 更新模型状态
            gr.update(value="本地模型")  # 确保模型选择按钮被选中
        )
    except Exception as e:
        print(f"❌ 处理模型文件失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return (
            gr.update(value=None),
            gr.update(value=f"❌ 处理模型文件失败: {str(e)}"),
            gr.update(value="yolo11n.pt")
        )


def update_yolo_model_selection(yolo_model_select_value):
    """更新YOLO模型选择和显示"""
    global current_yolo_model_path

    print(f"🔄 更新模型选择: {yolo_model_select_value}")

    # 处理不同的模型选择
    if yolo_model_select_value == "yolo11n.pt":
        model_path = Config.YOLO_MODEL_PATH
        current_yolo_model_path = model_path
        if os.path.exists(model_path):
            model_info = f"**默认模型:** yolo11n.pt"
        else:
            model_info = "❌ **默认模型 yolo11n.pt 不存在，请将模型文件放置在 models/ 目录下**"
        return (
            gr.update(visible=True),  # yolo_model_select 保持可见
            gr.update(visible=False),  # 隐藏本地模型上传组件
            gr.update(value=model_info)  # 更新模型信息
        )
    elif yolo_model_select_value == "本地模型":
        # 显示本地模型上传组件
        # 检查是否有已经上传的本地模型
        if current_yolo_model_path and os.path.exists(current_yolo_model_path):
            filename = os.path.basename(current_yolo_model_path)
            model_info = f"**当前模型:** {filename}"
        else:
            model_info = "**请上传本地.pt模型文件**"

        return (
            gr.update(visible=True),  # yolo_model_select 保持可见
            gr.update(visible=True),  # 显示本地模型上传组件
            gr.update(value=model_info)  # 更新模型信息
        )
    else:
        # 如果传入了其他值，保持当前状态
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(
                value=f"**模型:** {os.path.basename(yolo_model_select_value) if isinstance(yolo_model_select_value, str) else '未知'}")
        )


def update_image_task_ui(task_subtype, yolo_model_select):
    """更新图像任务UI组件"""
    global current_yolo_model_path

    # 清理任务类型（移除表情符号）
    cleaned_subtype = clean_task_subtype(task_subtype)

    # 更新模型选择组件的可见性
    if cleaned_subtype in ["目标检测", "图像分类", "实例分割"]:
        yolo_select_visible = True
        # 只有当选择了"本地模型"时才显示上传组件
        local_upload_visible = (yolo_model_select == "本地模型")
    else:
        yolo_select_visible = False
        local_upload_visible = False

    # 更新模型信息显示
    if yolo_model_select == "yolo11n.pt":
        model_info_text = f"**任务:** {cleaned_subtype} | **模型:** yolo11n.pt"
        model_status_text = "默认模型"
    elif yolo_model_select == "本地模型" and current_yolo_model_path and os.path.exists(current_yolo_model_path):
        filename = os.path.basename(current_yolo_model_path)
        model_info_text = f"**任务:** {cleaned_subtype} | **模型:** {filename}"
        model_status_text = "本地模型成功加载"
    else:
        model_info_text = f"**任务:** {cleaned_subtype} | **模型:** 未选择"
        model_status_text = "未选择"

    return (
        gr.update(visible=yolo_select_visible),  # yolo_model_select
        gr.update(visible=local_upload_visible),  # local_model_upload_group
        gr.update(value=model_info_text),  # model_info
        gr.update(value=model_status_text)  # model_status_text
    )


def collect_files(files):
    """收集所有文件，支持文件和文件夹"""
    all_files = []

    if not files:
        return []

    for file_info in files:
        file_path = file_info.name

        # 如果是文件夹，遍历所有图片文件
        if os.path.isdir(file_path):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
            for root, dirs, walk_files in os.walk(file_path):
                for file in walk_files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        all_files.append({
                            'name': os.path.join(root, file),
                            'is_file': True
                        })
        # 如果是文件
        else:
            all_files.append({
                'name': file_path,
                'is_file': True
            })

    return all_files


def process_files(files, task_type, model_type, api_key, preset_type,
                  custom_system_prompt, custom_task_prompt, concurrency,
                  text_input=None, vision_model="",
                  task_subtype="🎯 目标检测", yolo_model_select="yolo11n.pt",
                  local_model_file=None, conf_threshold=0.25):
    """处理文件的主函数 - 返回生成器"""

    global current_results, current_task_type, current_file_paths, current_visualizations, current_detections, current_original_images, current_editing_mode, current_edit_image_index, current_edit_annotations, current_task_subtype, current_yolo_model_path

    # 重置全局变量
    current_results = []
    current_file_paths = []
    current_visualizations = []
    current_detections = {}
    current_original_images = {}
    current_editing_mode = False
    current_edit_image_index = -1
    current_edit_annotations = []
    current_task_type = task_type

    # 清理任务类型字符串
    cleaned_task_subtype = clean_task_subtype(task_subtype)
    current_task_subtype = cleaned_task_subtype if task_type == "图像任务" else ""

    # 处理文本输入的情况
    if task_type == "问答对生成" and text_input and text_input.strip():
        # 优先处理文本输入
        if not api_key.strip():
            yield "请输入API密钥", None, None, None, None, gr.update(visible=False)
            return

        handler = APIHandler(model_type, api_key.strip())
        results = []

        # 获取提示词
        if preset_type == "自定义提示词":
            system_prompt = custom_system_prompt or "你是一个工业知识专家"
        else:
            preset = PRESETS.get(preset_type, {})
            system_prompt = preset.get("system_prompt", "你是一个工业知识专家")

        # 处理文本输入
        chunks = [{"text": text_input, "source_file": "文本输入", "page": 1}]

        for j, chunk in enumerate(chunks):
            qa_pairs = handler.generate_qa(chunk["text"], system_prompt)

            for qa in qa_pairs:
                result = {
                    "id": f"{len(results) + 1:04d}",
                    "task_type": preset_type,
                    "system_prompt": system_prompt,
                    "instruction": qa.get("instruction", ""),
                    "output": qa.get("output", ""),
                    "source_file": "文本输入",
                    "page_num": 1
                }
                results.append(result)

            yield f"正在处理文本段落 {j + 1}/{len(chunks)}，已生成 {len(results)} 条数据", None, None, None, None, gr.update(
                visible=False)

        # 保存结果
        if results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(Config.OUTPUT_DIR, f"text_qa_results_{timestamp}.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            final_msg = f"✅ 完成！已生成 {len(results)} 条QA数据\n已保存到: {output_path}"
            df_results = pd.DataFrame(results)
            current_results = results
            yield final_msg, df_results, None, None, None, gr.update(visible=True)
        else:
            yield "❌ 未能生成任何数据，请检查文本内容或API配置", None, None, None, None, gr.update(visible=False)

        return

    # 收集所有文件
    all_file_infos = collect_files(files)

    if not all_file_infos:
        yield "请先上传文件或输入文本", None, None, None, None, gr.update(visible=False)
        return

    results = []
    visualizations = []
    file_paths = []
    current_image_index = 0

    try:
        if task_type == "问答对生成":
            # ... [原有的问答对生成代码保持不变] ...
            if not api_key.strip():
                yield "请输入API密钥", None, None, None, None, gr.update(visible=False)
                return

            handler = APIHandler(model_type, api_key.strip())
            total_files = len(all_file_infos)

            for i, file_info in enumerate(all_file_infos):
                file_path = file_info['name']
                filename = os.path.basename(file_path)
                file_paths.append(file_path)
                status_msg = f"正在处理文件 {i + 1}/{total_files}: {filename}"
                yield status_msg, None, None, None, None, gr.update(visible=False)

                chunks = PDFProcessor.extract_text(file_path)

                if preset_type == "自定义提示词":
                    system_prompt = custom_system_prompt or "你是一个工业知识专家"
                else:
                    preset = PRESETS.get(preset_type, {})
                    system_prompt = preset.get("system_prompt", "你是一个工业知识专家")

                total_chunks = len(chunks)
                for j, chunk in enumerate(chunks):
                    progress_msg = f"{status_msg}\n正在处理段落 {j + 1}/{total_chunks}..."
                    yield progress_msg, None, None, None, None, gr.update(visible=False)

                    qa_pairs = handler.generate_qa(chunk["text"], system_prompt)

                    for qa in qa_pairs:
                        result = {
                            "id": f"{len(results) + 1:04d}",
                            "task_type": preset_type,
                            "system_prompt": system_prompt,
                            "instruction": qa.get("instruction", ""),
                            "output": qa.get("output", ""),
                            "source_file": chunk["source_file"],
                            "page_num": chunk["page"]
                        }
                        results.append(result)

                    time.sleep(0.1)

                yield f"已完成文件 {filename}，已生成 {len(results)} 条数据", None, None, None, None, gr.update(
                    visible=False)

            if results:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(Config.OUTPUT_DIR, f"qa_results_{timestamp}.json")

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                final_msg = f"✅ 完成！已生成 {len(results)} 条QA数据\n已保存到: {output_path}"
                df_results = pd.DataFrame(results)
                current_results = results
                current_file_paths = file_paths
                yield final_msg, df_results, None, None, None, gr.update(visible=True)
            else:
                yield "❌ 未能生成任何数据，请检查文件内容或API配置", None, None, None, None, gr.update(visible=False)

        elif task_type == "图像任务":
            # 确定模型路径
            model_path = None

            if yolo_model_select == "yolo11n.pt":
                model_path = Config.YOLO_MODEL_PATH
            elif yolo_model_select == "本地模型":
                # 使用全局变量中的模型路径
                model_path = current_yolo_model_path
                # 如果当前没有本地模型，尝试从上传的文件中获取
                if model_path is None and local_model_file is not None:
                    if hasattr(local_model_file, 'name'):
                        model_path = local_model_file.name
                    else:
                        model_path = str(local_model_file)
            else:
                # 直接使用传递的路径
                model_path = yolo_model_select

            # 如果模型路径为空，使用默认模型
            if model_path is None:
                model_path = Config.YOLO_MODEL_PATH
                print(f"⚠️ 使用默认模型: {model_path}")

            print(f"图像任务: 子类型={cleaned_task_subtype}, 模型={model_path}, 阈值={conf_threshold}")

            # 检查模型文件是否存在
            if not os.path.exists(model_path):
                yield f"❌ 模型文件不存在: {os.path.basename(model_path) if model_path else '未知'}", None, None, None, None, gr.update(
                    visible=False)
                return

            # 获取YOLO检测器
            detector = get_yolo_detector(model_path, cleaned_task_subtype)
            if detector is None:
                yield "❌ YOLO检测器初始化失败，请检查模型文件", None, None, None, None, gr.update(visible=False)
                return

            total_files = len(all_file_infos)

            for i, file_info in enumerate(all_file_infos):
                file_path = file_info['name']
                filename = os.path.basename(file_path)
                file_paths.append(file_path)

                status_msg = f"正在标注图片 {i + 1}/{total_files}: {filename}"
                yield status_msg, None, None, None, None, gr.update(visible=False)

                # 执行不同类型的图像任务
                if cleaned_task_subtype == "目标检测":
                    detections = detector.detect(file_path, conf_threshold=conf_threshold)
                elif cleaned_task_subtype == "图像分类":
                    detections = detector.classify(file_path, conf_threshold=conf_threshold)
                elif cleaned_task_subtype == "实例分割":
                    detections = detector.segment(file_path, conf_threshold=conf_threshold)
                else:
                    detections = []

                # 保存原始图像和检测结果
                original_image = Image.open(file_path)
                current_original_images[filename] = original_image
                current_detections[filename] = detections

                # 生成可视化图片
                vis_path = os.path.join(Config.VISUALIZATION_DIR, f"vis_{filename}")
                vis_img = detector.visualize(file_path, detections, vis_path)

                # 保存标注结果
                if detections:
                    # 根据任务类型选择保存格式
                    if cleaned_task_subtype == "目标检测":
                        # YOLO格式保存
                        yolo_txt_path = os.path.join(
                            Config.ANNOTATION_DIR,
                            f"{Path(filename).stem}.txt"
                        )
                        detector.save_yolo_format(filename, detections, yolo_txt_path)

                        # COCO格式保存
                        coco_json_path = os.path.join(
                            Config.ANNOTATION_DIR,
                            f"{Path(filename).stem}_coco.json"
                        )
                        detector.save_coco_format(filename, detections, coco_json_path)
                    elif cleaned_task_subtype == "图像分类":
                        # 分类结果保存为JSON
                        class_json_path = os.path.join(
                            Config.ANNOTATION_DIR,
                            f"{Path(filename).stem}_classification.json"
                        )
                        detector.save_classification_format(filename, detections, class_json_path)
                    elif cleaned_task_subtype == "实例分割":
                        # 分割结果保存
                        seg_json_path = os.path.join(
                            Config.ANNOTATION_DIR,
                            f"{Path(filename).stem}_segmentation.json"
                        )
                        detector.save_segmentation_format(filename, detections, seg_json_path)

                    # 为每个检测结果创建记录
                    for j, det in enumerate(detections):
                        if cleaned_task_subtype == "目标检测":
                            result = {
                                "id": f"{len(results) + 1:04d}",
                                "task_type": "图像任务-目标检测",
                                "image_file": filename,
                                "class_id": det.get('class_id', 0),
                                "class_name": det.get('class_name', 'unknown'),
                                "confidence": f"{det.get('confidence', 0):.4f}",
                                "bbox_xyxy": f"{det.get('bbox', [0, 0, 0, 0])[0]:.0f},{det.get('bbox', [0, 0, 0, 0])[1]:.0f},{det.get('bbox', [0, 0, 0, 0])[2]:.0f},{det.get('bbox', [0, 0, 0, 0])[3]:.0f}",
                                "bbox_yolo": f"{det.get('yolo_bbox', [0, 0, 0, 0])[0]:.6f},{det.get('yolo_bbox', [0, 0, 0, 0])[1]:.6f},{det.get('yolo_bbox', [0, 0, 0, 0])[2]:.6f},{det.get('yolo_bbox', [0, 0, 0, 0])[3]:.6f}",
                                "annotation_file": f"{Path(filename).stem}.txt",
                                "visualization": f"vis_{filename}"
                            }
                        elif cleaned_task_subtype == "图像分类":
                            result = {
                                "id": f"{len(results) + 1:04d}",
                                "task_type": "图像任务-图像分类",
                                "image_file": filename,
                                "class_id": det.get('class_id', 0),
                                "class_name": det.get('class_name', 'unknown'),
                                "confidence": f"{det.get('confidence', 0):.4f}",
                                "top_n": det.get('top_n', 5),
                                "annotation_file": f"{Path(filename).stem}_classification.json",
                                "visualization": f"vis_{filename}"
                            }
                        elif cleaned_task_subtype == "实例分割":
                            result = {
                                "id": f"{len(results) + 1:04d}",
                                "task_type": "图像任务-实例分割",
                                "image_file": filename,
                                "class_id": det.get('class_id', 0),
                                "class_name": det.get('class_name', 'unknown'),
                                "confidence": f"{det.get('confidence', 0):.4f}",
                                "bbox_xyxy": f"{det.get('bbox', [0, 0, 0, 0])[0]:.0f},{det.get('bbox', [0, 0, 0, 0])[1]:.0f},{det.get('bbox', [0, 0, 0, 0])[2]:.0f},{det.get('bbox', [0, 0, 0, 0])[3]:.0f}",
                                "mask_points": len(det.get('mask', [])) if det.get('mask') else 0,
                                "annotation_file": f"{Path(filename).stem}_segmentation.json",
                                "visualization": f"vis_{filename}"
                            }
                        results.append(result)

                    # 添加可视化图片到返回列表
                    if vis_img is not None:
                        # 转换BGR到RGB
                        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                        # 转换为PIL图像
                        pil_img = Image.fromarray(vis_img_rgb)
                        # 添加标题
                        caption = f"{filename} ({cleaned_task_subtype}: {len(detections)}个结果)"
                        visualizations.append((pil_img, caption))

                yield f"已标注 {i + 1}/{total_files} 张图片，检测到 {len(detections)} 个目标", None, visualizations, 0, total_files, gr.update(
                    visible=True)
                time.sleep(0.5)

            # 保存汇总结果
            if results:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                summary_path = os.path.join(Config.OUTPUT_DIR, f"annotation_summary_{timestamp}.json")

                summary = {
                    "task_type": cleaned_task_subtype,
                    "total_images": total_files,
                    "total_detections": len(results),
                    "model_used": model_path,
                    "conf_threshold": conf_threshold,
                    "detections_by_class": {},
                    "annotations": results
                }

                # 按类别统计
                for result in results:
                    class_name = result['class_name']
                    summary["detections_by_class"][class_name] = \
                        summary["detections_by_class"].get(class_name, 0) + 1

                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)

                df_results = pd.DataFrame(results)
                final_msg = f"✅ 完成！{cleaned_task_subtype} - 已标注 {total_files} 张图片，检测到 {len(results)} 个结果\n模型: {os.path.basename(model_path) if model_path else '默认'}\n已保存到: {summary_path}"
                current_results = results
                current_file_paths = file_paths
                current_visualizations = visualizations
                yield final_msg, df_results, visualizations, 0, total_files, gr.update(visible=True)
            else:
                yield f"❌ 未能检测到任何目标 (任务: {cleaned_task_subtype})", None, None, None, None, gr.update(
                    visible=False)

        else:  # 多模态生成
            # ... [原有的多模态生成代码保持不变] ...
            if not api_key.strip():
                yield "请输入API密钥", None, None, None, None, gr.update(visible=False)
                return

            handler = APIHandler(model_type, api_key.strip())
            total_files = len(all_file_infos)

            for i, file_info in enumerate(all_file_infos):
                file_path = file_info['name']
                filename = os.path.basename(file_path)
                file_paths.append(file_path)

                status_msg = f"正在分析图片 {i + 1}/{total_files}: {filename}"
                yield status_msg, None, None, None, None, gr.update(visible=False)

                analysis = handler.analyze_image(file_path)

                result = {
                    "id": f"{len(results) + 1:04d}",
                    "task_type": "图片理解",
                    "system_prompt": "你是一个工业图片分析专家",
                    "instruction": analysis.get("instruction", "请描述这张图片的内容"),
                    "image_file": filename,
                    "output": analysis.get("output", "分析失败")
                }
                results.append(result)

                progress = ((i + 1) / total_files) * 100
                yield f"已分析 {i + 1}/{total_files} 张图片，进度: {progress:.1f}%", None, None, None, None, gr.update(
                    visible=False)
                time.sleep(0.5)

            if results:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(Config.OUTPUT_DIR, f"image_results_{timestamp}.json")

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                final_msg = f"✅ 完成！已分析 {len(results)} 张图片\n已保存到: {output_path}"
                df_results = pd.DataFrame(results)
                current_results = results
                current_file_paths = file_paths
                yield final_msg, df_results, None, None, None, gr.update(visible=True)
            else:
                yield "❌ 未能分析图片，请检查文件或API配置", None, None, None, None, gr.update(visible=False)

    except Exception as e:
        error_msg = f"❌ 处理过程中发生错误: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        yield error_msg, None, None, None, None, gr.update(visible=False)


def save_edited_results(results_df, task_type):
    """保存编辑后的结果"""
    global current_results

    try:
        if results_df is None or len(results_df) == 0:
            return "❌ 没有可保存的数据", None

        # 转换DataFrame为字典列表
        edited_results = results_df.to_dict('records')

        # 保存编辑后的结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if task_type == "问答对生成":
            output_path = os.path.join(Config.OUTPUT_DIR, f"edited_qa_results_{timestamp}.json")
        elif task_type == "图像任务":
            output_path = os.path.join(Config.OUTPUT_DIR, f"edited_annotation_results_{timestamp}.json")
        else:  # 多模态生成
            output_path = os.path.join(Config.OUTPUT_DIR, f"edited_image_results_{timestamp}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(edited_results, f, ensure_ascii=False, indent=2)

        # 更新当前结果
        current_results = edited_results

        return f"✅ 编辑后的结果已保存到: {output_path}", output_path
    except Exception as e:
        return f"❌ 保存失败: {str(e)}", None


def navigate_images(direction, current_index, total_images):
    """导航图片"""
    global current_editing_mode, current_edit_image_index

    if total_images == 0:
        return 0, None, None, None, None, gr.update(visible=False), gr.update(visible=False)

    new_index = current_index + direction

    if new_index < 0:
        new_index = total_images - 1
    elif new_index >= total_images:
        new_index = 0

    # 退出编辑模式
    if current_editing_mode:
        current_editing_mode = False
        current_edit_image_index = -1

    # 获取当前图片
    if current_visualizations and len(current_visualizations) > new_index:
        vis_data = current_visualizations[new_index]
        if isinstance(vis_data, tuple):
            current_image, caption = vis_data

            # 获取当前图片的文件名
            if current_file_paths and len(current_file_paths) > new_index:
                current_file = current_file_paths[new_index]
                filename = os.path.basename(current_file)

                # 获取当前图片的检测结果
                image_results = [r for r in current_results if r.get('image_file') == filename]
                df_image_results = pd.DataFrame(image_results) if image_results else pd.DataFrame()

                return new_index, [(current_image,
                                    caption)], f"图片 {new_index + 1}/{total_images}: {filename}", df_image_results, gr.update(
                    visible=True), gr.update(visible=False)

    return new_index, None, f"图片 {new_index + 1}/{total_images}", None, gr.update(visible=False), gr.update(
        visible=False)


def toggle_edit_mode(current_index, total_images):
    """切换编辑模式"""
    global current_editing_mode, current_edit_image_index, current_edit_annotations

    if total_images == 0 or current_index < 0:
        return gr.update(visible=False), "❌ 没有图片可编辑"

    current_editing_mode = not current_editing_mode

    if current_editing_mode:
        current_edit_image_index = current_index

        # 获取当前图片
        if current_file_paths and len(current_file_paths) > current_index:
            current_file = current_file_paths[current_index]
            filename = os.path.basename(current_file)

            # 获取当前图片的检测结果
            image_results = [r for r in current_results if r.get('image_file') == filename]
            current_edit_annotations = image_results.copy()

            return gr.update(visible=True), f"📝 进入编辑模式: {filename}\n双击坐标进行修改，按Enter保存"
    else:
        return gr.update(visible=False), "退出编辑模式"


def update_annotation(image_index, annotations_df):
    """根据编辑的表格更新标注"""
    global current_results, current_visualizations

    if image_index < 0 or not current_file_paths or len(current_file_paths) <= image_index:
        return None, None, "❌ 更新失败: 无效的图片索引"

    try:
        file_path = current_file_paths[image_index]
        filename = os.path.basename(file_path)

        # 加载原始图像
        original_image = Image.open(file_path)
        draw = ImageDraw.Draw(original_image)

        # 更新当前结果
        updated_results = []

        # 保留不是当前图片的其他结果
        for result in current_results:
            if result.get('image_file') != filename:
                updated_results.append(result)

        # 处理编辑后的结果
        if annotations_df is not None and len(annotations_df) > 0:
            for _, row in annotations_df.iterrows():
                try:
                    # 创建新结果
                    new_result = {
                        "id": str(row.get('ID', f"{len(updated_results) + 1:04d}")),
                        "task_type": "图像任务",
                        "image_file": filename,
                        "class_id": get_class_id(str(row.get('类别', ''))),
                        "class_name": str(row.get('类别', '')),
                        "confidence": str(row.get('置信度', '0.80')),
                        "bbox_xyxy": str(row.get('坐标(x1,y1,x2,y2)', '0,0,100,100')),
                        "bbox_yolo": convert_xyxy_to_yolo(
                            list(map(float, str(row.get('坐标(x1,y1,x2,y2)', '0,0,100,100')).split(','))),
                            original_image.width,
                            original_image.height
                        ),
                        "annotation_file": f"{Path(filename).stem}_edited.txt",
                        "visualization": f"vis_{filename}",
                        "is_edited": True
                    }
                    updated_results.append(new_result)

                    # 绘制框
                    bbox_str = str(row.get('坐标(x1,y1,x2,y2)', '0,0,100,100'))
                    if bbox_str:
                        bbox = list(map(int, map(float, bbox_str.split(','))))
                        draw.rectangle(bbox, outline='red', width=3)
                        label = f"{row.get('类别', '')} {row.get('置信度', '')}"
                        draw.text((bbox[0], bbox[1] - 15), label, fill='red')
                except Exception as e:
                    print(f"处理行时出错: {e}")
                    continue

        # 更新全局变量
        current_results = updated_results

        # 更新可视化图像
        caption = f"{filename} (已编辑，{len([r for r in updated_results if r.get('image_file') == filename])}个目标)"
        if current_visualizations and len(current_visualizations) > image_index:
            current_visualizations[image_index] = (original_image, caption)

        # 更新数据框
        image_results = [r for r in updated_results if r.get('image_file') == filename]
        df_image_results = pd.DataFrame(image_results) if image_results else pd.DataFrame()

        return [(original_image, caption)], df_image_results, f"✅ 标注已更新: {filename}"

    except Exception as e:
        return None, None, f"❌ 更新失败: {str(e)}"


def add_new_annotation(image_index, class_name, confidence):
    """添加新的标注"""
    global current_edit_annotations

    if image_index < 0 or not current_file_paths or len(current_file_paths) <= image_index:
        return gr.update(), "❌ 添加失败: 无效的图片索引"

    try:
        # 创建新的标注
        new_annotation = {
            "id": f"{len(current_edit_annotations) + 1:04d}",
            "图像文件": os.path.basename(current_file_paths[image_index]),
            "类别": class_name,
            "置信度": f"{confidence:.2f}",
            "坐标(x1,y1,x2,y2)": "0,0,100,100"  # 默认坐标，用户可以在表格中修改
        }

        # 添加到当前编辑的标注列表
        current_edit_annotations.append(new_annotation)

        # 更新表格
        df_annotations = pd.DataFrame(current_edit_annotations)

        return df_annotations, f"✅ 已添加新标注: {class_name}"

    except Exception as e:
        return gr.update(), f"❌ 添加失败: {str(e)}"


def get_class_id(class_name):
    """根据类别名称获取类别ID"""
    # 简化版本，实际应该从YOLO模型中获取
    class_mapping = {
        "person": 0, "cat": 15, "dog": 16, "car": 2, "bicycle": 1,
        "motorcycle": 3, "bus": 5, "truck": 7, "bird": 14, "horse": 17
    }
    return class_mapping.get(class_name.lower(), 0)


def convert_xyxy_to_yolo(bbox_xyxy, img_width, img_height):
    """将XYXY坐标转换为YOLO格式"""
    x1, y1, x2, y2 = bbox_xyxy

    # 计算中心点坐标
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height

    # 计算宽度和高度
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return f"{x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f}"


def save_current_annotations():
    """保存当前标注结果"""
    global current_results

    try:
        if not current_results:
            return "❌ 没有可保存的标注数据", None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(Config.OUTPUT_DIR, f"interactive_annotations_{timestamp}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)

        return f"✅ 交互式标注结果已保存到: {output_path}", output_path
    except Exception as e:
        return f"❌ 保存失败: {str(e)}", None


def create_interface():
    """创建Gradio界面"""
    css = """
    .gradio-container {
        background: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=2072&q=80') center/cover no-repeat fixed !important;
        min-height: 100vh !important;
        padding: 20px !important;
        font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
    }
    .gradio-container::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.6);
        z-index: -1;
    }
    .gradio-container > * {
        position: relative;
        z-index: 1;
    }
    .main-header {
        text-align: center;
        font-size: 48px !important;
        font-weight: bold !important;
        color: white !important;
        margin-bottom: 20px !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        padding: 25px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        border-radius: 20px;
        border: 3px solid rgba(255, 255, 255, 0.2);
    }
    .app-subtitle {
        text-align: center;
        font-size: 20px !important;
        color: #e0e0e0 !important;
        margin-bottom: 40px !important;
        line-height: 1.6;
        padding: 0 20px;
    }
    .gr-box {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        margin-bottom: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
        backdrop-filter: blur(10px);
        /* 确保所有gr-box垂直顶部对齐 */
        vertical-align: top !important;
    }
    .annotation-editor {
        background: rgba(255, 255, 255, 0.98) !important;
        border: 2px solid #667eea !important;
        border-radius: 15px !important;
        padding: 20px !important;
        margin-top: 20px !important;
    }
    .edit-controls {
        background: rgba(255, 255, 255, 0.98) !important;
        border: 2px solid #28a745 !important;
        border-radius: 15px !important;
        padding: 15px !important;
        margin-bottom: 15px !important;
    }
    .gr-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
        margin: 5px !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2) !important;
    }
    .gr-button-edit {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%) !important;
    }
    .gr-button-cancel {
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%) !important;
    }
    .download-section {
        max-width: 600px !important;
        margin: 0 auto !important;
    }
    .right-column {
        margin-top: 0 !important;
    }
    .edit-table {
        max-height: 300px !important;
        overflow-y: auto !important;
        border: 1px solid #ddd !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }
    .model-upload-group {
        background: rgba(255, 255, 255, 0.98) !important;
        border: 2px dashed #667eea !important;
        border-radius: 10px !important;
        padding: 20px !important;
        margin-top: 10px !important;
    }
    .task-subtype-buttons {
        display: flex !important;
        justify-content: space-between !important;
        margin-bottom: 15px !important;
    }
    .task-subtype-btn {
        flex: 1 !important;
        margin: 0 5px !important;
    }
    """

    with gr.Blocks(title="山河智能数据标注工具") as app:
        # 页面标题
        gr.HTML("""
        <div class="main-header">
            🏭 山河智能数据标注工具
        </div>
        <div class="app-subtitle">
            支持问答对生成、图像任务、多模态分析 | 图像任务支持目标检测、图像分类、实例分割
        </div>
        """)

        # 状态变量
        task_type_state = gr.State("问答对生成")
        current_image_index = gr.State(0)
        total_images = gr.State(0)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="gr-box"):
                    # 工作模式
                    task_type = gr.Radio(
                        choices=["问答对生成", "图像任务", "多模态生成"],
                        label="📋 工作模式",
                        value="问答对生成"
                    )

                    # 模型选择
                    model_type = gr.Dropdown(
                        choices=list(API_CONFIG.keys()),
                        label="🤖 AI模型",
                        value="DeepSeek",
                        visible=True
                    )

                    # 多模态模型
                    vision_model = gr.Textbox(
                        label="多模态模型名称",
                        placeholder="例如：gpt-4-vision-preview",
                        visible=False
                    )

                    # API密钥
                    api_key = gr.Textbox(
                        label="🔑 API密钥",
                        placeholder="请输入API密钥",
                        type="password",
                        visible=True
                    )

                    # 测试连接
                    test_btn = gr.Button("📡 测试连接", variant="primary", visible=True)
                    test_output = gr.Textbox(label="连接状态", interactive=False, visible=True)

                    # YOLO模型配置
                    with gr.Group(visible=False) as yolo_config_group:
                        gr.Markdown("**图像任务配置**")

                        # 任务子类型选择
                        task_subtype = gr.Radio(
                            choices=["🎯 目标检测", "🖼️ 图像分类", "🖌️ 实例分割"],
                            label="任务类型",
                            value="🎯 目标检测",
                            elem_classes="task-subtype-buttons"
                        )

                        # 模型选择
                        yolo_model_select = gr.Radio(
                            choices=["yolo11n.pt", "本地模型"],
                            label="选择模型",
                            value="yolo11n.pt",
                            visible=True
                        )

                        # 本地模型上传
                        with gr.Group(visible=False) as local_model_upload_group:
                            local_model_file = gr.File(
                                label="上传本地模型文件",
                                file_types=[".pt"],
                                elem_classes="model-upload-group"
                            )
                            local_model_status = gr.Textbox(
                                label="模型状态",
                                interactive=False,
                                visible=True
                            )

                        # 置信度阈值
                        conf_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.25,
                            step=0.05,
                            label="置信度阈值"
                        )

                        # 模型信息显示
                        model_info = gr.Markdown("**模型:** yolo11n.pt | **任务:** 目标检测")

                    # 预设场景
                    preset_type = gr.Dropdown(
                        choices=list(PRESETS.keys()) + ["自定义提示词"],
                        label="📋 预设场景",
                        value="能碳知识查询",
                        visible=True
                    )

                    # 自定义提示词
                    with gr.Group(visible=True) as custom_prompt_group:
                        custom_system_prompt = gr.Textbox(
                            label="系统提示词",
                            placeholder="例如：你是一个能碳领域分析师...",
                            lines=2
                        )
                        custom_task_prompt = gr.Textbox(
                            label="任务提示词",
                            placeholder="例如：请生成碳核算相关的问答对...",
                            lines=2
                        )

                    # 并发数
                    concurrency = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="⚡ 并发请求数",
                        visible=True
                    )

            with gr.Column(scale=2, elem_classes="right-column"):
                # 文本输入区域
                with gr.Group(visible=True) as text_input_group:
                    text_input = gr.Textbox(
                        label="📝 输入文本内容（可选）",
                        placeholder="请在这里粘贴或输入文本内容...",
                        lines=8,
                        max_lines=20,
                        elem_id="text-input"
                    )

                # 统一文件上传区域
                with gr.Group(elem_classes="gr-box"):
                    gr.Markdown("### 📁 上传文件或文件夹")
                    file_upload = gr.File(
                        label="拖拽文件或文件夹到此处",
                        file_count="multiple",
                        file_types=[".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".gif", "directory"],
                        elem_id="unified-upload"
                    )
                    gr.Markdown("*支持上传单个文件、多个文件或整个文件夹*")

                # 处理按钮
                process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

                # 状态显示
                status_display = gr.Textbox(
                    label="📊 处理状态",
                    lines=4,
                    interactive=False
                )

                # 图像任务可视化区域
                with gr.Group(visible=False) as visualization_group:
                    gr.Markdown("### 🖼️ 检测可视化结果")

                    # 编辑控制按钮
                    with gr.Row():
                        edit_toggle_btn = gr.Button("✏️ 进入编辑模式", variant="primary", elem_classes="gr-button-edit",
                                                    visible=True)
                        edit_status = gr.Textbox(label="编辑状态", interactive=False, visible=False, lines=2)

                    # 编辑控制面板
                    with gr.Group(visible=False) as edit_control_group:
                        gr.Markdown("### 编辑控制")
                        with gr.Row():
                            new_class_name = gr.Textbox(
                                label="新增类别名称",
                                placeholder="输入新目标的类别名称",
                                value="person"
                            )
                            new_confidence = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.8,
                                step=0.1,
                                label="置信度"
                            )
                            add_annotation_btn = gr.Button("➕ 添加新标注", variant="secondary")

                        add_status = gr.Textbox(label="添加状态", interactive=False, visible=False)

                    # 可视化展示
                    visualization_gallery = gr.Gallery(
                        label="检测结果预览",
                        columns=2,
                        show_label=True,
                        height=400
                    )

                    # 图片导航区域
                    with gr.Group(visible=True) as image_navigation_group:
                        gr.Markdown("### 图片导航")
                        with gr.Row():
                            prev_btn = gr.Button("⬅️ 上一张", variant="secondary")
                            image_counter = gr.Textbox(
                                label="当前图片",
                                value="0/0",
                                interactive=False
                            )
                            next_btn = gr.Button("➡️ 下一张", variant="secondary")
                            update_btn = gr.Button("🔄 更新图像", variant="secondary", visible=False)

                    # 当前图片标注详情
                    current_image_results = gr.DataFrame(
                        label="当前图片标注详情（可编辑）",
                        headers=["ID", "图像文件", "类别", "置信度", "坐标(x1,y1,x2,y2)"],
                        datatype=["str", "str", "str", "str", "str"],
                        interactive=True,
                        wrap=True,
                        visible=False
                    )

                # 结果表格
                results_table = gr.DataFrame(
                    label="📋 生成结果（可编辑）",
                    headers=["ID", "任务类型", "图像/源文件", "类别/问题", "置信度/回答"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=True,
                    wrap=True,
                    visible=True
                )

        # 下载区域
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(elem_classes="gr-box download-section"):
                    gr.Markdown("### 结果操作")
                    with gr.Row():
                        save_btn = gr.Button("💾 保存编辑结果", variant="success")
                        save_interactive_btn = gr.Button("💾 保存交互式标注", variant="primary", visible=False)
                        download_btn = gr.Button("📥 下载结果文件", variant="secondary")

                    save_status = gr.Textbox(
                        label="保存状态",
                        interactive=False,
                        visible=False
                    )

        # 事件处理
        # 任务类型变化
        task_type.change(
            update_ui_components,
            inputs=[task_type],
            outputs=[
                model_type, api_key, test_btn, test_output,
                yolo_config_group, preset_type, concurrency,
                visualization_group, image_navigation_group,
                task_type_state, vision_model, custom_prompt_group,
                text_input_group, edit_toggle_btn, edit_control_group
            ]
        ).then(
            update_file_upload,
            inputs=[task_type],
            outputs=[file_upload]
        ).then(
            lambda x: gr.update(visible=True if x == "图像任务" else False),
            inputs=[task_type],
            outputs=[save_interactive_btn]
        )

        # 图像任务子类型变化
        task_subtype.change(
            update_image_task_ui,
            inputs=[task_subtype, yolo_model_select],
            outputs=[yolo_model_select, local_model_upload_group, model_info, local_model_status]
            # 添加 local_model_status
        )

        # YOLO模型选择变化
        yolo_model_select.change(
            update_yolo_model_selection,
            inputs=[yolo_model_select],
            outputs=[yolo_model_select, local_model_upload_group, model_info]
        )

        # 本地模型上传
        local_model_file.change(
            handle_local_model_upload,
            inputs=[local_model_file],
            outputs=[local_model_file, local_model_status, yolo_model_select]
        ).then(
            # 上传后更新任务UI，使用当前的子任务类型
            update_image_task_ui,
            inputs=[task_subtype, yolo_model_select],
            outputs=[yolo_model_select, local_model_upload_group, model_info, local_model_status]
        )


        # 测试连接
        test_btn.click(
            test_api_connection,
            inputs=[api_key, model_type],
            outputs=test_output
        )

        # 处理文件
        process_btn.click(
            process_files,
            inputs=[
                file_upload, task_type, model_type, api_key, preset_type,
                custom_system_prompt, custom_task_prompt, concurrency, text_input, vision_model,
                task_subtype, yolo_model_select, local_model_file, conf_threshold
            ],
            outputs=[status_display, results_table, visualization_gallery,
                     current_image_index, total_images, save_btn]
        )

        # 图片导航
        prev_btn.click(
            navigate_images,
            inputs=[gr.State(-1), current_image_index, total_images],
            outputs=[current_image_index, visualization_gallery, image_counter,
                     current_image_results, edit_control_group, update_btn]
        )

        next_btn.click(
            navigate_images,
            inputs=[gr.State(1), current_image_index, total_images],
            outputs=[current_image_index, visualization_gallery, image_counter,
                     current_image_results, edit_control_group, update_btn]
        )

        # 切换编辑模式
        edit_toggle_btn.click(
            toggle_edit_mode,
            inputs=[current_image_index, total_images],
            outputs=[edit_control_group, edit_status]
        )

        # 添加新标注
        add_annotation_btn.click(
            add_new_annotation,
            inputs=[current_image_index, new_class_name, new_confidence],
            outputs=[current_image_results, add_status]
        )

        # 更新图像
        update_btn.click(
            update_annotation,
            inputs=[current_image_index, current_image_results],
            outputs=[visualization_gallery, current_image_results, status_display]
        )

        # 保存编辑结果
        save_btn.click(
            save_edited_results,
            inputs=[results_table, task_type_state],
            outputs=[save_status, download_btn]
        )

        # 保存交互式标注
        save_interactive_btn.click(
            save_current_annotations,
            outputs=[save_status, download_btn]
        )

        # 下载结果
        download_btn.click(
            lambda x: x,
            inputs=[download_btn],
            outputs=gr.File(label="下载文件")
        )

    return app, css


if __name__ == "__main__":
    # 启动应用
    app, css = create_interface()

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        debug=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=css
    )