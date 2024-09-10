import os
from rembg import remove
from PIL import Image

def remove_background(input_path, output_path):
    with open(input_path, 'rb') as i:
        input_image = i.read()

    output_image = remove(input_image)

    with open(output_path, 'wb') as o:
        o.write(output_image)

input_folder = 'imagenes_fondo'
output_folder = 'imagen_sin_fondo'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f'{filename}')
        remove_background(input_path, output_path)
