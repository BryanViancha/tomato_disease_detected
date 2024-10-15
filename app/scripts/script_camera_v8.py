import cv2
from ultralytics import YOLO

# Variables para IP, usuario y contraseña
ip_camera = '192.168.0.100'  # Cambia la IP cuando sea necesario
user = 'admin'
password = 'supercamera123'

# Crear la URL RTSP usando las variables
rtsp_url = f'rtsp://{user}:{password}@{ip_camera}:554/user={user}_password={password}_channel=1_stream=0.sdp?real_stream'

# Cargar un modelo YOLOv8 ligero para mejorar rendimiento
model = YOLO('../../runs/train/model1_v8/weights/best.pt')  # Cambia a yolov8s.pt si se necesita mayor precisión, pero con menos rendimiento

# Configurar para usar GPU si está disponible
# model.to('cuda')  # Si no tienes GPU, puedes comentar esta línea/

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("No se puede acceder a la cámara RTSP. Verifica la URL y la conexión.")
    exit()

print("Conexión exitosa a la cámara. Presiona 'q' para salir.")

# Reducir la resolución para optimizar rendimiento
desired_width = 720
desired_height = 540

# Procesar solo cada N frames para mejorar el rendimiento
frame_count = 0
skip_frames = 2  # Procesar solo cada tercer frame

# Bucle para capturar y procesar frames en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el frame.")
        break

    # Redimensionar el frame para procesar menos datos y acelerar el análisis
    frame_resized = cv2.resize(frame, (desired_width, desired_height))

    # Procesar solo cada N frames para reducir la carga
    if frame_count % skip_frames == 0:
        # Realizar predicción con un umbral de confianza mínimo para mejorar la precisión
        results = model(frame_resized, conf=0.5)  # conf=0.5 asegura que solo se muestren detecciones de alta confianza

        # Filtrar las detecciones para mostrar solo "minador" y "alternaria" usando OpenCV para mayor velocidad
        for result in results:
            for box in result.boxes:
                cls = box.cls[0].item()
                name = model.names[int(cls)]
                if name in ['minador', 'alternaria']:
                    # Extraer las coordenadas y convertirlas a enteros
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                    color = (255, 0, 0) if name == 'alternaria' else (
                        0, 0, 255)  # Rojo para alternaria, azul para minador

                    # Dibujar el cuadro delimitador y la etiqueta en el frame
                    cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame_resized, f"{name} ({box.conf[0].item():.2f})",
                                (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el frame procesado con OpenCV para mejorar la visualización en tiempo real
    cv2.imshow("Detección en tiempo real con YOLOv8", frame_resized)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Incrementar el contador de frames

# Liberar recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
