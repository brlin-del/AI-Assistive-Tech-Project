import cv2
from ultralytics import YOLO

class VisionAssistant:
    def __init__(self, model_path='weights/yolov8n.pt'):
        print("[INFO] Loading AI Model from weights folder...")
        self.model = YOLO(model_path)
        
    def detect_objects(self, frame):
        results = self.model(frame, stream=True)
        detected_items = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                confidence = float(box.conf[0])
                if confidence > 0.50:
                    detected_items.append(label)
                    
        return list(set(detected_items))

if __name__ == "__main__":
    assistant = VisionAssistant()
    print("[SUCCESS] AI Model Loaded. Ready to process mobile camera streams!")
