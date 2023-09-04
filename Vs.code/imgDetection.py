import cv2
import math
from ultralytics import YOLO

model = YOLO('best.pt')
classNames = ['Blackspot', 'Canker', 'Fresh', 'Grenning']
# image_path = 'fresh.jpg'  # Coloque o caminho para sua imagem
#image_path = 'canker.jpg'
#image_path = 'blackspot.jpg'
#image_path = 'grenning.jpg'
#image_path = 'teste1.jpeg'
image_path = 'teste2.jpeg'
#image_path = 'teste3.jpeg'

frame = cv2.imread(image_path)
frame = cv2.resize(frame, (640, 640))
results = model(frame)

boxes = results[0].boxes

for box in boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # confidence
    confidence = math.ceil((box.conf[0]*100))/100
    print("Confidence --->", confidence)

    # class name
    cls = int(box.cls[0])
    print("Class name -->", classNames[cls])

    # object details
    org = [x1, y1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    cv2.putText(frame, str(
        classNames[cls] + ' ' + str(confidence)), org, font, fontScale, color, thickness)

cv2.imshow('Image Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
