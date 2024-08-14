import os
from rembg import remove
from PIL import Image


def remove_background(input_path, output_path):
    # Abrir la imagen
    with open(input_path, 'rb') as i:
        input_image = i.read()

    # Eliminar el fondo
    output_image = remove(input_image)

    # Guardar la imagen resultante
    with open(output_path, 'wb') as o:
        o.write(output_image)


# Ruta de la carpeta de imágenes y la carpeta de salida
input_folder = 'imagenes_fondo'
output_folder = 'imagen_sin_fondo'

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Procesar todas las imágenes en la carpeta
for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'output_{filename}')
        remove_background(input_path, output_path)
