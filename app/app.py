import os
import time
from datetime import datetime, timedelta
from threading import Timer
import cv2
from dotenv import load_dotenv
from twilio.rest import Client
from ultralytics import YOLO

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
TARGET_PHONE_NUMBER = os.getenv('TARGET_PHONE_NUMBER')
IP_CAMERA = os.getenv('IP_CAMERA')
USER = os.getenv('CAMERA_USER')
PASSWORD = os.getenv('CAMERA_PASSWORD')
RTSP_URL = f'rtsp://{USER}:{PASSWORD}@{IP_CAMERA}:554/user={USER}_password={PASSWORD}_channel=1_stream=0.sdp?real_stream'
MODEL_PATH = os.getenv('MODEL_PATH')

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

detection_counter = {}


def send_detection_summary():
    global detection_counter
    message = "Resumen de detecciones en las últimas 24 horas:\n"
    if not detection_counter:
        message += "- No se detectaron enfermedades\n"
    else:
        for disease, count in detection_counter.items():
            message += f"- {disease}: {count} detección(es)\n"

    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=TARGET_PHONE_NUMBER
        )
        print(f"Mensaje resumen enviado: {message}")
    except Exception as e:
        print(f"Error al enviar el resumen: {e}")

    detection_counter = {}


def send_daily_check():
    try:
        client.messages.create(
            body="Recordatorio diario: La vigilancia del invernadero está activa.",
            from_=TWILIO_PHONE_NUMBER,
            to=TARGET_PHONE_NUMBER
        )
        print("Mensaje diario enviado.")
    except Exception as e:
        print(f"Error al enviar el recordatorio: {e}")


def schedule_daily_summary():
    now = datetime.now()
    target_time = now.replace(hour=17, minute=0, second=0, microsecond=0)
    if now > target_time:
        target_time += timedelta(days=1)
    delay = (target_time - now).total_seconds()
    Timer(delay, send_detection_summary).start()


def schedule_daily_check():
    now = datetime.now()
    target_time = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now > target_time:
        target_time += timedelta(days=1)
    delay = (target_time - now).total_seconds()
    Timer(delay, send_daily_check).start()


def connect_camera(url, retries=3):
    for attempt in range(retries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print("Conexión exitosa a la cámara.")
            return cap
        print(f"Intento {attempt + 1} fallido. Reintentando...")
        time.sleep(2)
    raise Exception("No se puede acceder a la cámara RTSP después de varios intentos.")


def process_frame(frame, model):
    frame_resized = cv2.resize(frame, (640, 420))
    results = model(frame_resized, conf=0.3)

    for result in results:
        for box in result.boxes:
            cls = box.cls[0].item()
            name = model.names[int(cls)]
            if name != 'hojaSana':
                if name not in detection_counter:
                    detection_counter[name] = 1
                else:
                    detection_counter[name] += 1
                return frame_resized, name, box.xyxy[0], box.conf[0].item()
    return frame_resized, None, None, None


def detection_loop(cap, model):
    frame_count = 0
    skip_frames = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al recibir el frame.")
            break

        if frame_count % skip_frames == 0:
            frame_resized, disease, coords, confidence = process_frame(frame, model)

            if disease and coords is not None:
                xmin, ymin, xmax, ymax = map(int, coords)
                color = (255, 0, 0) if disease == 'alternaria' else (0, 0, 255)
                cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame_resized, f"{disease} ({confidence:.2f})",
                            (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Detección en tiempo real con YOLOv8", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1


if __name__ == "__main__":
    try:
        schedule_daily_summary()
        schedule_daily_check()

        cap = connect_camera(RTSP_URL)

        model = YOLO(MODEL_PATH)

        detection_loop(cap, model)

    except (cv2.error, Exception) as e:
        print(f"Error: {e}")

    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
