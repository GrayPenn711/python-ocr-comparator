import easyocr
import time

# 全局初始化 reader，避免重复初始化
reader = None

def get_reader():
    global reader
    if reader is None:
        print("    正在初始化 EasyOCR...")
        reader = easyocr.Reader(['ch_sim'], gpu=False)
    return reader

def recognize_text(image_path):
    reader = get_reader()
    
    start_time = time.time()
    results = reader.readtext(image_path)
    end_time = time.time()
    
    texts = []
    boxes = []
    confidences = []
    
    for (bbox, text, confidence) in results:
        texts.append(text)
        boxes.append([int(x) for point in bbox for x in point])
        confidences.append(confidence)
    
    return {
        'text': '\n'.join(texts),
        'boxes': boxes,
        'confidences': confidences,
        'time': end_time - start_time
    }