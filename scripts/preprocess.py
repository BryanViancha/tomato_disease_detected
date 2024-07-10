import os
import cv2
import numpy as np
from xml.etree import ElementTree


def preprocess_image(image_path):
    # Lee una imagen, la redimensiona a 224x224
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalizar los valores de los pixeles a [0, 1]
    return image


def load_data(data_directory):
    # Cargar las imagenes y las etiquetadas desde el directorio donde se encuentran
    images = []
    labels = []

    for filename in os.listdir(data_directory):  # Recorrer todos los archivos en el directorio
        if filename.endswith('.jpg'):  # Filtrar solo los archivos JPEG
            image_path = os.path.join(data_directory, filename)
            label_path = os.path.join(data_directory, filename.replace('.jpg', '.xml'))
            if not os.path.exists(label_path):
                print(f"Etiqueta no encontrada para la imagen {image_path}. Esperado: {label_path}")
                continue  # Saltar esta imagen si el archivo XML no existe
            tree = ElementTree.parse(label_path)  # Parsear el archivo XML
            root = tree.getroot()
            xml_filename = root.find('filename').text
            if xml_filename != filename:
                print(f"Nombre del archivo en XML no coincide: {xml_filename} vs {filename}")
                continue  # Saltar esta imagen si el nombre del archivo no coincide
            image = preprocess_image(image_path)  # Preprocesar la imagen
            label = 0  # Por defecto, la etiqueta es 0 (sano)
            for obj in root.findall('object'):
                if obj.find('name').text == 'minadorDisease':  # Si la etiqueta es 'diseased', se asigna 1
                    label = 1
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)  # Convertir listas a arrays de Numpy
