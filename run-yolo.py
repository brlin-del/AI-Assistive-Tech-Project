import sys
sys.path.append('venv/lib/python3.14/site-packages')
from ultralytics import YOLO

model = YOLO ('wieghts/yolov8n.pt')
modle.predict('bus.jpg , save=True')
