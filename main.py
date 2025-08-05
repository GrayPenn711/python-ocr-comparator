import os
import json
from pathlib import Path
import cv2
import numpy as np

# 尝试导入各个OCR引擎，失败的会跳过
ocr_modules = {}
try:
    from ocr_engines.easy_ocr import recognize_text as easy_ocr_recognize
    ocr_modules["EasyOCR"] = easy_ocr_recognize
except ImportError as e:
    print(f"Warning: EasyOCR not available: {e}")
try:
    from ocr_engines.paddle_ocr import recognize_text as paddle_ocr_recognize
    ocr_modules["PaddleOCR"] = paddle_ocr_recognize
except ImportError as e:
    print(f"Warning: PaddleOCR not available: {e}")

try:
    from ocr_engines.tesseract_ocr import recognize_text as tesseract_recognize
    ocr_modules["Tesseract"] = tesseract_recognize
except ImportError as e:
    print(f"Warning: Tesseract not available: {e}")

try:
    from ocr_engines.vision_ocr import recognize_text as vision_ocr_recognize
    ocr_modules["VisionOCR"] = vision_ocr_recognize
except ImportError as e:
    print(f"Warning: VisionOCR not available: {e}")

try:
    from ocr_engines.rapid_ocr import recognize_text as rapid_ocr_recognize
    ocr_modules["RapidOCR"] = rapid_ocr_recognize
except ImportError as e:
    print(f"Warning: RapidOCR not available: {e}")


def generate_result_image(image_path, result, output_dir):
    """生成带标记识别区域的结果图片"""
    try:
        # 读取原图
        image = cv2.imread(str(image_path))
        if image is None:
            return
        
        # 绘制识别区域
        boxes = result.get('boxes', [])
        for i, box in enumerate(boxes):
            if len(box) >= 8:  # 确保有足够的坐标点
                # 将坐标点重塑为多边形格式
                points = np.array([[box[j], box[j+1]] for j in range(0, len(box), 2)], np.int32)
                
                # 绘制多边形边框
                cv2.polylines(image, [points], True, (0, 255, 0), 2)
                
                # 在左上角添加序号
                if len(points) > 0:
                    cv2.putText(image, str(i+1), tuple(points[0]), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 保存结果图片
        output_image_path = output_dir / f"{image_path.stem}_result.png"
        cv2.imwrite(str(output_image_path), image)
        
    except Exception as e:
        print(f"    生成结果图片失败: {e}")

def main():
    images_dir = Path("data/images")
    results_dir = Path("data/results")
    
    # 创建结果目录
    for engine in ["EasyOCR", "PaddleOCR", "Tesseract", "VisionOCR", "RapidOCR"]:
        (results_dir / engine).mkdir(parents=True, exist_ok=True)
    
    # 检查是否有可用的OCR引擎
    if not ocr_modules:
        print("Error: No OCR engines are available!")
        return
    
    # 获取所有图片
    image_files = [f for f in images_dir.glob("*") if f.suffix.lower() == '.png']
    
    # 循环处理每张图片
    for image_path in image_files:
        print(f"\nProcessing: {image_path.name}")
        
        # 循环调用每个OCR
        for engine_name, ocr_func in ocr_modules.items():
            try:
                result = ocr_func(str(image_path))
                
                # 转换为数组格式的 JSON
                texts = result['text'].split('\n')
                boxes = result.get('boxes', [])
                
                # 创建数组格式的结果
                json_result = []
                for i, text in enumerate(texts):
                    if text.strip() and i < len(boxes):
                        json_result.append({
                            "序号": i + 1,
                            "识别内容": text,
                            "坐标": boxes[i]
                        })
                
                # 保存数组格式的结果
                output_file = results_dir / engine_name / f"{image_path.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(json_result, f, ensure_ascii=False, indent=2)
                
                # 生成带标记的结果图片
                generate_result_image(image_path, result, results_dir / engine_name)
                
                # 输出识别结果的文字和坐标
                print(f"  {engine_name} 识别结果:")
                texts = result['text'].split('\n')
                boxes = result.get('boxes', [])
                
                # 显示每个识别的文字和对应坐标
                for i, text in enumerate(texts):
                    if text.strip() and i < len(boxes):
                        coord = boxes[i]
                        print(f"    [{i+1}] \"{text}\" → 坐标: {coord[:8]}")  # 只显示前8个坐标值
                
                # 如果识别结果太多，只显示前10个
                if len(texts) > 10:
                    print(f"    ... 共识别 {len(texts)} 个文本区域")
                
            except Exception as e:
                print(f"  {engine_name}: Error - {str(e)}")


if __name__ == "__main__":
    main()