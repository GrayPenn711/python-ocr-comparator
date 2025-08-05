from paddleocr import PaddleOCR
import time

def recognize_text(image_path):
    # 使用新的参数名
    ocr = PaddleOCR(use_textline_orientation=True, lang='ch')
    
    start_time = time.time()
    # 使用新的 predict 方法
    result = ocr.predict(image_path)
    end_time = time.time()
    
    texts = []
    boxes = []
    confidences = []
    
    # 处理新版本的 OCRResult 对象
    if result and len(result) > 0:
        ocr_result = result[0]  # 第一页结果
        
        # 获取文本行数据
        for i in range(len(ocr_result)):
            line_data = ocr_result[i]
            
            # 提取文本
            if hasattr(line_data, 'text'):
                text = line_data.text
            else:
                continue
                
            # 提取坐标 (bbox)
            if hasattr(line_data, 'bbox'):
                bbox = line_data.bbox
                # bbox 通常是 [x1, y1, x2, y2] 格式，转换为四个角点
                boxes.append([
                    int(bbox[0]), int(bbox[1]),    # 左上角
                    int(bbox[2]), int(bbox[1]),    # 右上角  
                    int(bbox[2]), int(bbox[3]),    # 右下角
                    int(bbox[0]), int(bbox[3])     # 左下角
                ])
            else:
                boxes.append([0, 0, 0, 0, 0, 0, 0, 0])
                
            # 提取置信度
            if hasattr(line_data, 'score'):
                confidence = float(line_data.score)
            else:
                confidence = 1.0
                
            texts.append(text)
            confidences.append(confidence)
    
    return {
        'text': '\n'.join(texts),
        'boxes': boxes,
        'confidences': confidences,
        'time': end_time - start_time
    }