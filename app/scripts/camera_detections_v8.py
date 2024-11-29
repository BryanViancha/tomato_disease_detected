# SCRIPT DE PRUEBA PARA CONSUMIR MODELO DE YOLOV8 MEDIANTE LA CAMARA
import time
import cv2
from ultralytics import YOLO

model = YOLO('../../runs/train/model1_v8/weights/best.pt')

colors = {
    'minador': (255, 0, 0),  # Azul
    'alternaria': (0, 0, 255)  # Rojo
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

execution_time = 50
start_time = time.time()

confidence_threshold = 0.5
iou_threshold = 0.45

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > execution_time:
        print("Tiempo de predicción completado.")
        break

    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo obtener el frame de la cámara.")
        break

    results = model.predict(source=frame, conf=confidence_threshold, iou=iou_threshold, save=False, show=False)

    for result in results[0].boxes:
        box = result.xyxy[0]
        conf = result.conf[0]
        cls = int(result.cls[0])
        xmin, ymin, xmax, ymax = map(int, box)

        class_name = model.names[cls]

        color = colors.get(class_name, (0, 255, 0))

        label = f"{class_name} {conf:.2f}"
        frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        frame = cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Texto

    cv2.imshow('YOLOv8 - Detección en Tiempo Real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
