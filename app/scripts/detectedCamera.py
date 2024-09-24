import cv2
import torch
import time

# Cargar el modelo YOLOv5
model = torch.hub.load('../../yolov5', 'custom', path='../../yolov5/runs/train/TomatoIA5/weights/best.pt',
                       source='local')

# Definir colores para las clases: minador (azul) y alternaria (rojo)
colors = {
    'minador': (255, 0, 0),  # Azul
    'alternaria': (0, 0, 255)  # Rojo
}

# Configurar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Ajustar la resolución de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Tiempo de ejecución en segundos
execution_time = 50  # Por ejemplo, 50 segundos
start_time = time.time()

# Umbral de confianza e IOU
confidence_threshold = 0.5

# Bucle de detección
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

    # Redimensionar el frame si es necesario
    # frame_resized = cv2.resize(frame, (640, 640))

    # Realizar las predicciones
    results = model(frame)  # Eliminar los parámetros conf e iou

    # Filtrar las predicciones por confianza
    filtered_results = []
    for result in results.xyxy[0]:  # results.xyxy[0] contiene las detecciones
        xmin, ymin, xmax, ymax, conf, cls = result[:6]

        # Filtrar por confianza
        if conf >= confidence_threshold:
            filtered_results.append(result)

    # Dibujar las predicciones filtradas en el frame
    for result in filtered_results:
        xmin, ymin, xmax, ymax, conf, cls = result[:6]
        class_name = model.names[int(cls)]

        # Asignar color según la clase
        color = colors.get(class_name, (0, 255, 0))  # Verde por defecto si la clase no es "minador" o "alternaria"

        label = f"{class_name} {conf:.2f}"
        frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)  # Dibujar rectángulo
        frame = cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Texto

    # Mostrar el frame con las detecciones
    cv2.imshow('YOLOv5 - Detección en Tiempo Real', frame)

    # Oprimir 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
