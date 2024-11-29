# SCRIPT PARA HACER PRUEBAS MEDIANTE ENVIO DE IMAGENES CON POSTMAN
# CONSUMIENDO LOS ENDPOINTS DE YOLOV5 Y YOLOV8
import io
import os
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

YOLOV5_PATH = os.path.join(os.path.dirname(__file__), '../yolov5')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../yolov5/runs/train/model1_v5/weights/best.pt')
modelv5 = torch.hub.load(YOLOV5_PATH, 'custom', path=MODEL_PATH, source='local')

modelv8 = YOLO('../../runs/train/model1_v8/weights/best.pt')

colors = {
    'minador': 'blue',
    'alternaria': 'red'
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

    output_path = os.path.join('images_output', 'image_output.png')
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


@app.route('/yolov5', methods=['POST'])
def predictV5():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
    file = request.files['file']
    try:
        file_path = os.path.join('images_input', file.filename)
        file.save(file_path)
        results = modelv5(file_path)
        predictions = []
        for result in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = result[:6]
            predictions.append({
                'bbox': [xmin.item(), ymin.item(), xmax.item(), ymax.item()],
                'confidence': conf.item(),
                'class': int(cls.item()),
                'name': modelv5.names[int(cls.item())]
            })
        output_image_path = visualize_detections(file_path, predictions)
        os.remove(file_path)
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/yolov8', methods=['POST'])
def predictV8():
    OUTPUT_FOLDER = 'images_output'

    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó ningún archivo'}), 400

    for filename in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    image_file = request.files['file']
    image = Image.open(io.BytesIO(image_file.read()))

    results = modelv8.predict(image)

    img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    predictions = []
    for result in results:
        df = result.boxes.data.cpu().numpy()
        for box in df:
            xmin, ymin, xmax, ymax, confidence, cls = box
            class_name = modelv8.names[int(cls)]

            if class_name in ['minador', 'alternaria']:
                color = (0, 0, 255) if class_name == 'alternaria' else (255, 0, 0)
                cv2.rectangle(img_cv2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 6)
                cv2.putText(img_cv2, f"{class_name} ({confidence:.2f})",
                            (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

                prediction = {
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax),
                    "confidence": float(confidence),
                    "class": int(cls),
                    "name": class_name
                }
                predictions.append(prediction)

    output_image_path = os.path.join(OUTPUT_FOLDER, "image_output.png")
    cv2.imwrite(output_image_path, img_cv2)

    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
