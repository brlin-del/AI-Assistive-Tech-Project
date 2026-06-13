import sys
sys.path.append('venv/lib/python3.14/site-packages')
from ultralytics import YOLO 

model = YOLO('weights/yolov8n.pt')
model.predict('bus.jpg', save=True)
