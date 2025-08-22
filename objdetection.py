import cv2
from ultralytics import YOLO

model = YOLO("potato.pt")

capdev = cv2.VideoCapture(0)

ret = True

while ret:
    ret, frame = capdev.read()
    if not ret:
        break
    
    result = model.predict(source=frame, imgsz=640, conf=0.5)
    annotate = result[0].plot()
    cv2.imshow("detect", annotate)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capdev.release()
cv2.destroyAllWindows()