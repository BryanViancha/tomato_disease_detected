import os

# Ruta de la carpeta de imágenes a renombrar
carpeta_entrada = '../../imagenes'

# Ruta de la carpeta de salida (donde se guardarán los archivos renombrados)
carpeta_salida = '../../imagenes_renombradas'

# Crear la carpeta de salida si no existe
if not os.path.exists(carpeta_salida):
    os.makedirs(carpeta_salida)

# Contador para numerar los archivos
contador = 1

# Procesar todas las imágenes de la carpeta de entrada
for archivo in os.listdir(carpeta_entrada):
    if archivo.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        # Ruta completa del archivo de entrada
        ruta_archivo = os.path.join(carpeta_entrada, archivo)

        # Crear el nuevo nombre de archivo con número secuencial
        nombre_base, extension = os.path.splitext(archivo)
        nuevo_nombre = f"{nombre_base}_{contador}{extension}"

        # Ruta completa del archivo de salida
        ruta_salida = os.path.join(carpeta_salida, nuevo_nombre)

        # Renombrar el archivo
        os.rename(ruta_archivo, ruta_salida)

        # Incrementar el contador
        contador += 1

print("Renombrado de archivos completado.")
