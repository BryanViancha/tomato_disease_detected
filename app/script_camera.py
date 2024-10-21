import cv2
from ultralytics import YOLO
from twilio.rest import Client
import time



client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Variables para IP, usuario y contraseña
ip_camera = '192.168.0.100'  # Cambia la IP cuando sea necesario
user = 'admin'
password = 'supercamera123'

# Crear la URL RTSP usando las variables
rtsp_url = f'rtsp://{user}:{password}@{ip_camera}:554/user={user}_password={password}_channel=1_stream=0.sdp?real_stream'

# Cargar un modelo YOLOv8
model = YOLO('../../runs/train/model1_v8/weights/best.pt')

# Lista de enfermedades a detectar
DISEASES = ['minador', 'alternaria']

# Intervalo mínimo entre alertas en segundos (ej: 1 hora = 3600 segundos)
alert_interval = 3600  # 1 hora entre alertas
last_alert_time = 0  # Tiempo del último envío de alerta

# Contador de detecciones
detection_counter = {
    'minador': 0,
    'alternaria': 0
}

# Función para enviar el mensaje
def send_message(disease):
    message = f'¡Alerta! Se ha detectado la enfermedad: {disease}. Veces detectada: {detection_counter[disease]}'
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=TARGET_PHONE_NUMBER
    )
    print(f"Mensaje enviado: {message}")

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("No se puede acceder a la cámara RTSP. Verifica la URL y la conexión.")
    exit()

print("Conexión exitosa a la cámara. Presiona 'q' para salir.")

# Reducir la resolución para optimizar rendimiento
desired_width = 720
desired_height = 540

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

    # Filtrar las detecciones para mostrar solo enfermedades
    for result in results:
        for box in result.boxes:
            cls = box.cls[0].item()
            name = model.names[int(cls)]
            if name in DISEASES:
                # Actualizar contador de detecciones
                detection_counter[name] += 1

                # Verificar si ha pasado suficiente tiempo desde la última alerta
                current_time = time.time()
                if current_time - last_alert_time > alert_interval:
                    send_message(name)
                    last_alert_time = current_time

                # Extraer las coordenadas y convertirlas a enteros
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                color = (255, 0, 0) if name == 'alternaria' else (0, 0, 255)  # Rojo para alternaria, azul para minador

                # Dibujar el cuadro delimitador y la etiqueta en el frame
                cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame_resized, f"{name} ({box.conf[0].item():.2f})",
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar el frame procesado
    cv2.imshow("Detección en tiempo real con YOLOv8", frame_resized)

    # Salir al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
