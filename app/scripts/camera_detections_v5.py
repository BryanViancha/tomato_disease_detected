# SCRIPT DE PRUEBA PARA CONSUMIR MODELO DE YOLOV5 MEDIANTE LA CAMARA
import time
import cv2
import torch

model = torch.hub.load('../../yolov5', 'custom', path='../../yolov5/runs/train/model1_v5/weights/best.pt',
                       source='local')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

execution_time = 50
start_time = time.time()

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

    results = model(frame)

    for result in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = result[:6]
        label = f"{model.names[int(cls)]} {conf:.2f}"
        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        frame = cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0),
                            2)

    cv2.imshow('YOLOv5 - Detección en Tiempo Real', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
