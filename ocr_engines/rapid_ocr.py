from rapidocr_onnxruntime import RapidOCR
import time

def recognize_text(image_path):
    engine = RapidOCR()
    
    start_time = time.time()
    result, elapse = engine(image_path, return_word_box=False)
    end_time = time.time()
    
    texts = []
    boxes = []
    confidences = []
    
    if result:
        for line in result:
            dt_boxes, rec_res, score = line
            texts.append(rec_res)
            # dt_boxes 可能已经是列表，不需要 reshape
            if hasattr(dt_boxes, 'reshape'):
                boxes.append([int(x) for x in dt_boxes.reshape(-1).tolist()])
            else:
                # 如果已经是列表，直接展平
                flat_box = []
                for point in dt_boxes:
                    flat_box.extend([int(point[0]), int(point[1])])
                boxes.append(flat_box)
            confidences.append(score)
    
    return {
        'text': '\n'.join(texts),
        'boxes': boxes,
        'confidences': confidences,
        'time': end_time - start_time
    }