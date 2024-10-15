import cv2
from ultralytics import YOLO

# Variables para IP, usuario y contraseña
ip_camera = '192.168.0.100'  # Cambia la IP cuando sea necesario
user = 'admin'
password = 'supercamera123'

# Crear la URL RTSP usando las variables
rtsp_url = f'rtsp://{user}:{password}@{ip_camera}:554/user={user}_password={password}_channel=1_stream=0.sdp?real_stream'
# rtsp_url = 'rtsp://192.168.0.103:554/user=admin_password=UxgudjsE_channel=1_stream=0.sdp?real_stream'

# Cargar el modelo YOLOv8
model = YOLO('../runs/train/model2_v8/weights/best.pt')

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("No se puede acceder a la cámara RTSP. Verifica la URL y la conexión.")
    exit()

print("Conexión exitosa a la cámara. Presiona 'q' para salir.")

# Dimensiones deseadas para la visualización del frame
desired_width = 740
desired_height = 580

# Bucle para capturar y procesar frames en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el frame.")
        break

    # Redimensionar el frame
    frame_resized = cv2.resize(frame, (desired_width, desired_height))

    # Realizar predicción con YOLOv8
    results = model(frame_resized)

    # Extraer y dibujar las predicciones en el frame
    annotated_frame = results[0].plot()

    # Mostrar el frame con las predicciones
    cv2.imshow("Detección en tiempo real con YOLOv8", annotated_frame)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
