import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv5 o YOLOv8 (ajusta la ruta según tu modelo)
model = YOLO('../runs/train/model1_v8/weights/best.pt')

# URL RTSP de la cámara (la que obtuviste)
rtsp_url = 'rtsp://camera_a572:Camera123:192.168.0.101:80/'

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("No se puede acceder a la cámara RTSP")
    exit()

# Procesar los frames de la cámara en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el frame")
        break

    # Realizar predicción con YOLOv5 o YOLOv8
    results = model(frame)

    # Mostrar los resultados en el frame
    annotated_frame = results[0].plot()

    # Mostrar el frame con las predicciones
    cv2.imshow("Detección en tiempo real", annotated_frame)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
