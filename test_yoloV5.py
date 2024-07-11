from flask import Flask, request, jsonify, send_file
from PIL import Image
import torch
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches

app = Flask(__name__)

# Configurar el registro de errores
logging.basicConfig(level=logging.DEBUG)

# Ruta al archivo de pesos del modelo
MODEL_PATH = 'yolov5/runs/train/exp/weights/best.pt'
# Ruta al repositorio YOLOv5 local
YOLOV5_PATH = 'yolov5'
model = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local')  # source='local' asegura que se use el repositorio clonado localmente


def visualize_detections(image_path, predictions):
    # Cargar la imagen
    img = Image.open(image_path)

    # Crear una figura y un eje
    fig, ax = plt.subplots()

    # Mostrar la imagen
    ax.imshow(img)

    # Dibujar las cajas delimitadoras
    for pred in predictions:
        xmin, ymin, xmax, ymax = pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, pred['name'], bbox=dict(facecolor='yellow', alpha=0.5))

    # Guardar la imagen con las detecciones
    output_path = os.path.join('uploads', 'detections.png')
    plt.savefig(output_path)
    plt.close(fig)

    return output_path

#Endpoint para realizar predicciones sobre imágenes de hojas.
@app.route('/testYoloV5', methods=['POST'])
def predict():
    if 'file' not in request.files:
        app.logger.error('No se proporcionó ningún archivo')
        return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

    file = request.files['file']
    if file.filename == '':
        app.logger.error('No se selecciono un archivo')
        return jsonify({'error': 'No se selecciono un archivo'}), 400

    try:
        # Guardar la imagen
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Realizar la predicción
        img = Image.open(file_path)
        results = model(img)

        # Obtener las predicciones en formato JSON
        predictions = results.pandas().xyxy[0].to_dict(orient="records")

        # Visualizar las detecciones
        output_path = visualize_detections(file_path, predictions)

        # Eliminar la imagen después de la predicción
        os.remove(file_path)

        # Devolver la imagen con las detecciones
        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        app.logger.error('Error during prediction: %s', e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
