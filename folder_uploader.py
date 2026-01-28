import os
import gradio as gr
from typing import List


def get_image_files_from_folder(folder_path: str) -> List[str]:
    """从文件夹中获取所有图片文件"""
    if not folder_path or not os.path.exists(folder_path):
        return []

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in image_extensions:
                image_files.append(os.path.join(root, file))

    return image_files


def create_folder_upload_component():
    """创建文件夹上传组件"""
    with gr.Group() as folder_upload_group:
        folder_path = gr.Textbox(
            label="📂 文件夹路径",
            placeholder="请输入或粘贴文件夹路径，或点击浏览按钮选择",
            interactive=True
        )

        browse_btn = gr.Button("浏览文件夹", variant="secondary", size="sm")

        # 文件列表显示
        file_list = gr.File(
            label="检测到的图片文件",
            file_count="multiple",
            visible=False
        )

        # 使用JavaScript来处理文件夹选择
        js_code = """
        function() {
            const input = document.createElement('input');
            input.type = 'file';
            input.webkitdirectory = true;
            input.multiple = true;

            input.onchange = function(e) {
                const files = Array.from(e.target.files);
                const folderPath = files.length > 0 ? files[0].webkitRelativePath.split('/')[0] : '';

                // 更新路径
                document.querySelector('[data-testid="textbox"]').value = folderPath;

                // 触发更新
                const event = new Event('input', { bubbles: true });
                document.querySelector('[data-testid="textbox"]').dispatchEvent(event);
            };

            input.click();
        }
        """

        browse_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=js_code
        )

    return folder_upload_group, folder_path, file_list