from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data="./open-images-v7/data.yaml", epochs=35 ,imgsz=640,batch=5)