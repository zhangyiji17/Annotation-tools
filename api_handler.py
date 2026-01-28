import requests
import json
import base64
from typing import List, Dict, Optional
import time
from config import API_CONFIG


class APIHandler:
    """API处理器"""

    def __init__(self, model_type: str, api_key: str):
        self.model_type = model_type
        self.api_key = api_key
        self.config = API_CONFIG.get(model_type, API_CONFIG["DeepSeek"])

    def test_connection(self) -> bool:
        """测试连接"""
        try:
            headers = self._get_headers()
            payload = {
                "model": self.config["default_model"],
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 5
            }

            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )

            return response.status_code == 200
        except Exception as e:
            print(f"连接测试失败: {str(e)}")
            return False

    def generate_qa(self, text: str, system_prompt: str, max_qa: int = 10) -> List[Dict]:
        """生成问答对"""
        try:
            headers = self._get_headers()

            user_prompt = f"""请从以下文本生成{max_qa}个独立问答对:
{text[:3000]}

要求:
1. 每个问答对格式为:
问题: [问题内容]
答案: [答案内容]

2. 问答对之间用空行分隔
3. 答案要专业、准确
"""

            payload = {
                "model": self.config["default_model"],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }

            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                return self._parse_qa_response(response.json())
            else:
                print(f"API错误: {response.status_code}, {response.text}")
                return []
        except Exception as e:
            print(f"生成QA失败: {str(e)}")
            return []

    def analyze_image(self, image_path: str) -> Dict:
        """分析图片"""
        try:
            if not self.config.get("supports_vision", False):
                return {
                    "instruction": "请描述这张图片的内容",
                    "output": "当前模型不支持图片分析"
                }

            # 读取图片
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()

            headers = self._get_headers()
            payload = {
                "model": self.config.get("default_vision_model", self.config["default_model"]),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请详细描述这张图片的内容"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }

            response = requests.post(
                f"{self.config['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                if "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                elif "result" in data:
                    content = data["result"]
                else:
                    content = str(data)

                return {
                    "instruction": "请描述这张图片的内容",
                    "output": content
                }
            else:
                return {
                    "instruction": "请描述这张图片的内容",
                    "output": f"分析失败: {response.status_code}"
                }
        except Exception as e:
            return {
                "instruction": "请描述这张图片的内容",
                "output": f"分析出错: {str(e)}"
            }

    def _parse_qa_response(self, data: Dict) -> List[Dict]:
        """解析QA响应"""
        try:
            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
            elif "result" in data:
                content = data["result"]
            else:
                return []

            qa_pairs = []
            lines = content.strip().split('\n')

            current_question = ""
            current_answer = ""

            for line in lines:
                line = line.strip()
                if line.startswith("问题:") or line.startswith("问题："):
                    if current_question and current_answer:
                        qa_pairs.append({
                            "instruction": current_question,
                            "output": current_answer
                        })
                    current_question = line.split(":", 1)[-1].strip()
                    current_answer = ""
                elif line.startswith("答案:") or line.startswith("答案："):
                    current_answer = line.split(":", 1)[-1].strip()
                elif current_question and line:
                    if current_answer:
                        current_answer += "\n" + line
                    else:
                        current_answer = line

            if current_question and current_answer:
                qa_pairs.append({
                    "instruction": current_question,
                    "output": current_answer
                })

            return qa_pairs[:10]  # 最多返回10个
        except Exception as e:
            print(f"解析响应失败: {str(e)}")
            return []

    def _get_headers(self) -> Dict:
        """获取请求头"""
        if self.model_type == "DeepSeek":
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        elif self.model_type == "文心一言":
            return {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        else:  # ChatGPT等
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }