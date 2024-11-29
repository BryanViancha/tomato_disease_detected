import time
import cv2
from onvif import ONVIFCamera
from ultralytics import YOLO

# ip_camera = '192.168.0.100'
ip_camera = '192.168.154.246'
user = 'admin'
password = 'supercamera123'
rtsp_url = f'rtsp://{user}:{password}@{ip_camera}:554/user={user}_password={password}_channel=1_stream=0.sdp?real_stream'

model = YOLO('../../runs/train/model1_v8/weights/best.pt')

camera = ONVIFCamera(ip_camera, 8899, user, password)
ptz = camera.create_ptz_service()
media_profile = camera.create_media_service().GetProfiles()[0]

def stop_camera():
    try:
        ptz.Stop({'ProfileToken': media_profile.token})
        print("Movimiento detenido.")
    except Exception as e:
        print(f"No se pudo detener el movimiento previo: {e}")

def reset_camera_position():
    request = ptz.create_type('AbsoluteMove')
    request.ProfileToken = media_profile.token
    request.Position = {'PanTilt': {'x': 0, 'y': 0}}
    ptz.AbsoluteMove(request)
    print("Cámara posicionada en el centro.")
    time.sleep(2)

stop_camera()
reset_camera_position()

move_right = True
delay_between_moves = 3
step = 0.05
max_steps = 10
barrido_repeticiones = 3

def move_camera(direction):
    request = ptz.create_type('ContinuousMove')
    request.ProfileToken = media_profile.token
    request.Velocity = {'PanTilt': {'x': step if direction == 'right' else -step, 'y': 0}}
    ptz.ContinuousMove(request)
    time.sleep(1)
    ptz.Stop({'ProfileToken': media_profile.token})

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("No se puede acceder a la cámara RTSP.")
    exit()

print("Iniciando el monitoreo y movimiento automático. Presiona 'q' para salir.")

steps = 0
barrido_actual = 0
desired_width = 480
desired_height = 360

while barrido_actual < barrido_repeticiones:
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir el frame.")
        break

    frame_resized = cv2.resize(frame, (desired_width, desired_height))

    results = model(frame_resized, conf=0.5)

    for result in results:
        for box in result.boxes:
            cls = box.cls[0].item()
            name = model.names[int(cls)]
            if name in ['minador', 'alternaria']:
                xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
                color = (255, 0, 0) if name == 'alternaria' else (0, 0, 255)
                cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame_resized, f"{name} ({box.conf[0].item():.2f})",
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detección en tiempo real con YOLOv8", frame_resized)

    if steps >= max_steps:
        move_right = not move_right
        steps = 0
        barrido_actual += 1
        print(f"Barrido {barrido_actual}/{barrido_repeticiones} completado.")

    move_camera('right' if move_right else 'left')
    steps += 1

    time.sleep(delay_between_moves)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
