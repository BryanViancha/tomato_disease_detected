import os
from PIL import Image

# Ruta de la carpeta de imágenes de entrada y la carpeta de salida
carpeta_entrada = '../../imagenes'
carpeta_salida = '../../imagenes_redimensionadas'

# Crear la carpeta de salida si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)


# Función para redimensionar la imagen manteniendo la relación de aspecto
def redimensionar_imagen(imagen, tamaño_maximo=2560):
    ancho, alto = imagen.size

    if ancho > alto:
        # Imagen horizontal
        nuevo_ancho = tamaño_maximo
        nuevo_alto = int(alto * (tamaño_maximo / ancho))
    else:
        # Imagen vertical
        nuevo_alto = tamaño_maximo
        nuevo_ancho = int(ancho * (tamaño_maximo / alto))

    print(f"Original: {ancho}x{alto} | Redimensionado: {nuevo_ancho}x{nuevo_alto}")  # Debug

    # Redimensionar la imagen
    imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

    # Convertir a RGBA si la imagen tiene un canal alfa para mantener la transparencia
    if imagen_redimensionada.mode != 'RGBA':
        imagen_redimensionada = imagen_redimensionada.convert('RGBA')

    return imagen_redimensionada


# Procesar todas las imágenes de la carpeta de entrada
for archivo in os.listdir(carpeta_entrada):
    if archivo.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        ruta_imagen = os.path.join(carpeta_entrada, archivo)
        imagen = Image.open(ruta_imagen)

        imagen_redimensionada = redimensionar_imagen(imagen)

        # Crear la ruta de salida
        ruta_salida = os.path.join(carpeta_salida, archivo)

        # Cambiar la extensión a .png si no está en formato PNG
        if not ruta_salida.lower().endswith('.png'):
            ruta_salida = os.path.splitext(ruta_salida)[0] + '.png'

        # Guardar la imagen en formato PNG para mantener la transparencia
        imagen_redimensionada.save(ruta_salida, format='PNG')

print("Proceso de redimensionado completado.")
