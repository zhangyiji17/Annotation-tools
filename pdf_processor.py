import fitz  # PyMuPDF
import re
import os
from typing import List, Dict


class PDFProcessor:
    """PDF处理器"""

    @staticmethod
    def extract_text(pdf_path: str, max_chars: int = 2000) -> List[Dict]:
        """提取PDF文本，分割成小块"""
        try:
            doc = fitz.open(pdf_path)
            chunks = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                if not text.strip():
                    continue

                # 按句子分割
                sentences = re.split(r'(?<=[。！？；\n])', text)

                current_chunk = ""
                for sentence in sentences:
                    if sentence.strip():
                        if len(current_chunk) + len(sentence) <= max_chars:
                            current_chunk += sentence
                        else:
                            if current_chunk:
                                chunks.append({
                                    "text": current_chunk.strip(),
                                    "page": page_num + 1,
                                    "source_file": os.path.basename(pdf_path)
                                })
                            current_chunk = sentence

                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page_num + 1,
                        "source_file": os.path.basename(pdf_path)
                    })

            return chunks
        except Exception as e:
            print(f"PDF处理错误: {str(e)}")
            return []