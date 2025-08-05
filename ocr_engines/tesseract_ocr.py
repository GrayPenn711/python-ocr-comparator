import pytesseract
from PIL import Image
import time

def recognize_text(image_path):
    start_time = time.time()
    
    image = Image.open(image_path)
    data = pytesseract.image_to_data(image, lang='chi_sim', output_type=pytesseract.Output.DICT)
    
    end_time = time.time()
    
    texts = []
    boxes = []
    confidences = []
    
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 0:
            text = data['text'][i].strip()
            if text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                texts.append(text)
                boxes.append([x, y, x + w, y, x + w, y + h, x, y + h])
                confidences.append(float(data['conf'][i]) / 100.0)
    
    return {
        'text': '\n'.join(texts),
        'boxes': boxes,
        'confidences': confidences,
        'time': end_time - start_time
    }