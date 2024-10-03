import time

import cv2
import torch

model = torch.hub.load('../../yolov5', 'custom', path='../../yolov5/runs/train/model1_v5/weights/best.pt',source='local')
# Abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

execution_time = 50
start_time = time.time()  # Tiempo inicial

# Bucle de detección
while True:
    current_time = time.time()  # Tiempo actual
    elapsed_time = current_time - start_time  # Tiempo transcurrido

    if elapsed_time > execution_time:
        print("Tiempo de predicción completado.")
        break

    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo obtener el frame de la cámara.")
        break

    # Realizar las predicciones en tiempo real
    results = model(frame)

    # Dibujar las predicciones en el frame
    for result in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = result[:6]
        label = f"{model.names[int(cls)]} {conf:.2f}"  # Etiqueta con la clase y confianza
        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)  # Rectángulo
        frame = cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0),
                            2)  # Texto

    # Mostrar el frame con las detecciones
    cv2.imshow('YOLOv5 - Detección en Tiempo Real', frame)

    # Oprime 'q' para salir antes de que el tiempo termine
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
