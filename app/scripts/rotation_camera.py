# SCRIPT PARA EJECUTAR MOVIENTO DE CAMARA PTZ
import time
from onvif import ONVIFCamera

ip_camera = '192.168.242.246'
user = 'admin'
password = 'supercamera123'
port = 8899

def connect_to_camera():
    camera = ONVIFCamera(ip_camera, port, user, password)
    ptz = camera.create_ptz_service()
    media_service = camera.create_media_service()
    profiles = media_service.GetProfiles()
    media_profile = profiles[0]
    return ptz, media_profile

def stop_camera(ptz, media_profile):
    try:
        ptz.Stop({'ProfileToken': media_profile.token})
        print("Movimiento detenido.")
        time.sleep(1)
    except Exception as e:
        print(f"Error al detener el movimiento: {e}")

def reset_camera_position(ptz, media_profile):
    try:
        request = ptz.create_type('AbsoluteMove')
        request.ProfileToken = media_profile.token
        request.Position = {'PanTilt': {'x': 0, 'y': 0}}
        ptz.AbsoluteMove(request)
        print("Cámara reposicionada al centro.")
        time.sleep(2)
    except Exception as e:
        print(f"Error al reposicionar la cámara: {e}")

ptz, media_profile = connect_to_camera()

stop_camera(ptz, media_profile)
reset_camera_position(ptz, media_profile)

def move_camera(ptz, media_profile, direction, duration=2):
    try:
        request = ptz.create_type('ContinuousMove')
        request.ProfileToken = media_profile.token
        request.Velocity = {'PanTilt': {'x': 0.5 if direction == 'right' else -0.5, 'y': 0}}
        ptz.ContinuousMove(request)
        time.sleep(duration)
        ptz.Stop({'ProfileToken': media_profile.token})
        print(f"Cámara movida hacia {direction}.")
    except Exception as e:
        print(f"Error al mover la cámara: {e}")

for _ in range(3):
    move_camera(ptz, media_profile, 'right', duration=2)
    time.sleep(1)
    move_camera(ptz, media_profile, 'left', duration=2)
    time.sleep(1)

print("Barrido completo.")
