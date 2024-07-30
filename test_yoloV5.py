import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return "La aplicación Flask está funcionando."

YOLOV5_PATH = os.path.join(os.path.dirname(__file__), 'yolov5')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'yolov5/runs/train/exp/weights/best.pt')

# Cargar el modelo YOLOv5
model = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local')

# def visualize_detections(image_path, predictions):
#     img = Image.open(image_path)
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#
#     for pred in predictions:
#         xmin, ymin, xmax, ymax = pred['bbox']
#         label = pred['name']
#         confidence = pred['confidence']
#         width, height = xmax - xmin, ymax - ymin
#         rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)
#         ax.text(xmin, ymin, f"{label} ({confidence:.2f})", bbox=dict(facecolor='yellow', alpha=0.5))
#
#     output_path = os.path.join('uploads', 'detections.png')
#     plt.savefig(output_path)
#     plt.close(fig)
#
#     return output_path
#
#
# @app.route('/testYoloV5', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400
#
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected'}), 400
#
#     try:
#         # Guardar el archivo
#         file_path = os.path.join('uploads', file.filename)
#         file.save(file_path)
#
#         # Realizar la predicción
#         results = model(file_path)
#
#         # Depuración: Verificar el contenido de results
#         print(results)
#
#         predictions = []
#         for result in results.xyxy[0]:  # results.xyxy[0] contiene las detecciones
#             xmin, ymin, xmax, ymax, conf, cls = result[:6]
#             predictions.append({
#                 'bbox': [xmin.item(), ymin.item(), xmax.item(), ymax.item()],
#                 'confidence': conf.item(),
#                 'class': int(cls.item()),
#                 'name': model.names[int(cls.item())]
#             })
#
#         # Eliminar el archivo temporal
#         os.remove(file_path)
#
#         return jsonify({'predictions': predictions})
#     except Exception as e:
#         # Depuración: Imprimir el error
#         print(f"Error: {e}")
#         return jsonify({'error': str(e)}), 500
#
#
# if __name__ == '__main__':
#     os.makedirs('uploads', exist_ok=True)
#     app.run(host='0.0.0.0', port=5000)

# Colores para cada clase
colors = {
    'minador': 'blue',
    'alternaria': 'red',
    'hojaSana': 'green'
}


def visualize_detections(image_path, predictions):
    img = Image.open(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)

    for pred in predictions:
        xmin, ymin, xmax, ymax = pred['bbox']
        label = pred['name']
        confidence = pred['confidence']
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=colors[label], facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"{label} ({confidence:.2f})", bbox=dict(facecolor='yellow', alpha=0.5))

    output_path = os.path.join('uploads', 'detections.png')
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


@app.route('/testYoloV5', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Guardar el archivo
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        # Realizar la predicción
        results = model(file_path)
        print('results => ', results)
        print('results => ', results.xyxy)

        predictions = []
        for result in results.xyxy[0]:  # results.xyxy[0] contiene las detecciones
            xmin, ymin, xmax, ymax, conf, cls = result[:6]
            predictions.append({
                'bbox': [xmin.item(), ymin.item(), xmax.item(), ymax.item()],
                'confidence': conf.item(),
                'class': int(cls.item()),
                'name': model.names[int(cls.item())]
            })

        # Visualizar detecciones
        output_image_path = visualize_detections(file_path, predictions)

        # Eliminar el archivo temporal
        os.remove(file_path)

        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
