from ultralytics import YOLO
import cv2
import math
import sys
model = YOLO("./runs/detect/train9/weights/last.pt")
classNames = ["Owl","Sheep"]
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2
try:
  img = cv2.imread(sys.argv[1])
except:
   print("file in argument doesnt exist")
   exit()
   
results = model(img,conf= 0.6)
for r in results:
    boxes = r.boxes
    for box in boxes:
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
      cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
      confidence = math.ceil((box.conf[0]*100))/100
      cls = int(box.cls[0])
      org = [x1, y1]
      img = cv2.putText(img, classNames[cls]+str(confidence), org, font, fontScale, color, thickness)
cv2.imshow(sys.argv[1],img)
cv2.waitKey(0)
