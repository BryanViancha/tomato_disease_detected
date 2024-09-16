import io
import os
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('runs/train/experiment_14/weights/best.pt')

OUTPUT_FOLDER = 'app/processed_images'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/yolov8', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No ha enviado una imagen!!"}), 400

    for filename in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    image_file = request.files['file']
    image = Image.open(io.BytesIO(image_file.read()))

    results = model.predict(image)

    output_image_path = os.path.join(OUTPUT_FOLDER, "processed_image.png")
    results[0].save(output_image_path)

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

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
