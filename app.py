from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model_path = 'models/tomato_disease_model_1.h5'

# Verificar si el modelo existe antes de cargarlo
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Modelo cargado desde {model_path}")
else:
    raise FileNotFoundError(f"Modelo no encontrado en {model_path}")


def preprocess_image(image):
    """
    Preprocesa una imagen para que sea compatible con el modelo.
    Redimensiona la imagen a 224x224 píxeles y normaliza los valores de los píxeles.
    """
    image = cv2.resize(image, (224, 224))  # Redimensionar la imagen a 224x224 píxeles
    image = image / 255.0  # Normalizar los valores de los píxeles a [0, 1]
    image = np.expand_dims(image, axis=0)  # Expandir las dimensiones para que sea compatible con el modelo
    return image


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones sobre imágenes de hojas.
    """
    file = request.files['file']  # Obtener el archivo de imagen desde la solicitud
    image = np.fromfile(file, np.uint8)  # Leer la imagen desde el archivo
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Decodificar la imagen
    image = preprocess_image(image)  # Preprocesar la imagen
    prediction = model.predict(image)  # Realizar la predicción
    result = {'minadorDisease': bool(prediction[0][0] > 0.5)}  # Convertir la predicción a un resultado booleano
    return jsonify(result)  # Devolver el resultado como una respuesta JSON


if __name__ == '__main__':
    app.run(debug=True)  # Ejecutar el servidor Flask
