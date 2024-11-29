# SCRIPT PARA REDIMENSIONAMIENTO DE IMAGENES
import os
from PIL import Image

input_folder = 'images'
output_folder = 'resize_images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def resize_image(image, max_size=2560):
    width, height = image.size

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    print(f"Original: {width}x{height} | Resized: {new_width}x{new_height}")

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if resized_image.mode != 'RGBA':
        resized_image = resized_image.convert('RGBA')

    return resized_image


for file in os.listdir(input_folder):
    if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        image_path = os.path.join(input_folder, file)
        image = Image.open(image_path)

        resized_image = resize_image(image)

        output_path = os.path.join(output_folder, file)

        if not output_path.lower().endswith('.png'):
            output_path = os.path.splitext(output_path)[0] + '.png'

        resized_image.save(output_path, format='PNG')

print("Redimensionamiento completado.")
