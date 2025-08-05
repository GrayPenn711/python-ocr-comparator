# OCR 对比工具

对比多个 OCR 引擎识别效果的 Python 工具。

## 快速开始

1. **安装**
```bash
pip install -r requirements.txt
```

2. **使用**
```bash
# 把图片放到 data/images/ 文件夹
python main.py
```

3. **查看结果**
   - 结果保存在 `data/results/` 各引擎文件夹中
   - 包含 JSON 文件（文字+坐标）和标注图片

## 支持的 OCR

- Tesseract - 开源 OCR
- EasyOCR - 深度学习 OCR
- PaddleOCR - 百度 OCR  
- RapidOCR - 轻量级 OCR
- VisionOCR - macOS 专用

## 注意

- Python 3.13 只支持 Tesseract、RapidOCR、VisionOCR
- 完整支持需要 Python 3.11 或 3.12