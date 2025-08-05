import platform
import time

if platform.system() == 'Darwin':
    try:
        import Vision
        from Quartz import CGImageSourceCreateWithURL, CGImageSourceCreateImageAtIndex
        from Foundation import NSURL
        VISION_AVAILABLE = True
    except ImportError:
        VISION_AVAILABLE = False
else:
    VISION_AVAILABLE = False

def recognize_text(image_path):
    if not VISION_AVAILABLE:
        return {
            'text': '',
            'boxes': [],
            'confidences': [],
            'time': 0,
            'error': 'Vision Framework not available'
        }
    
    start_time = time.time()
    
    url = NSURL.fileURLWithPath_(image_path)
    image_source = CGImageSourceCreateWithURL(url, None)
    cg_image = CGImageSourceCreateImageAtIndex(image_source, 0, None)
    
    # 获取图像尺寸
    import Quartz
    img_width = Quartz.CGImageGetWidth(cg_image)
    img_height = Quartz.CGImageGetHeight(cg_image)
    
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    # 设置识别语言为中文
    request.setRecognitionLanguages_(['zh-Hans'])
    
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
    handler.performRequests_error_([request], None)
    
    end_time = time.time()
    
    texts = []
    boxes = []
    confidences = []
    
    observations = request.results()
    if observations:
        for observation in observations:
            text = observation.text()
            confidence = observation.confidence()
            bbox = observation.boundingBox()
            
            # Vision Framework 使用归一化坐标 (0-1)，需要转换为像素坐标
            # 注意：Vision 的 y 坐标从底部开始，需要翻转
            x = bbox.origin.x * img_width
            y = (1.0 - bbox.origin.y - bbox.size.height) * img_height
            w = bbox.size.width * img_width
            h = bbox.size.height * img_height
            
            texts.append(text)
            # 转换为四个角点的坐标格式
            boxes.append([
                int(x), int(y),           # 左上角
                int(x + w), int(y),       # 右上角
                int(x + w), int(y + h),   # 右下角
                int(x), int(y + h)        # 左下角
            ])
            confidences.append(float(confidence))
    
    return {
        'text': '\n'.join(texts),
        'boxes': boxes,
        'confidences': confidences,
        'time': end_time - start_time
    }