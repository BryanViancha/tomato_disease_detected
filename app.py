from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os

# Inicializa la aplicación Flask
app = Flask(__name__)

# Carga el modelo YOLOv8
model = YOLO('runs/train/experiment_14/weights/best.pt')  # Ajusta la ruta a tu modelo entrenado si es necesario

# Carpeta donde se guardarán las imágenes procesadas
OUTPUT_FOLDER = 'analisis'

# Asegúrate de que la carpeta de análisis existe
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


# Define la ruta para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request   .files:
        return jsonify({"error": "No image provided"}), 400

    # Elimina cualquier archivo previo en la carpeta de análisis
    for filename in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Lee la imagen desde la solicitud
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))

    # Realiza la predicción
    results = model.predict(image)

    # Guardar la imagen procesada en la carpeta de análisis
    output_image_path = os.path.join(OUTPUT_FOLDER, "processed_image.png")
    results[0].save(output_image_path)  # Esto guarda la imagen con las cajas dibujadas

    # Preparar las predicciones en formato JSON
    predictions = []
    for result in results:
        df = result.boxes.data.cpu().numpy()
        for box in df:
            xmin, ymin, xmax, ymax, confidence, cls = box
            prediction = {
                "xmin": float(xmin),
                "ymin": float(ymin),
                "xmax": float(xmax),
                "ymax": float(ymax),
                "confidence": float(confidence),
                "class": int(cls),
                "name": model.names[int(cls)]
            }
            predictions.append(prediction)

    # Retornar las predicciones como respuesta JSON
    return jsonify(predictions)


# Ejecuta la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
