import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    results = model(frame)

    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        

    cv2.imshow('Webcam Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()