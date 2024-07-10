import cv2
import numpy as np

# Cargar la red y los parámetros
net = cv2.dnn.readNet("backup/yolov3_final.weights", "../cfg/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
classes = ["hojaSana", "minador", "alternaria"]

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects(img):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs

def get_boxes(outs, height, width):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return class_ids, confidences, boxes

def draw_labels(img, class_ids, confidences, boxes):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0) if label == "hojaSana" else (0, 0, 255) if label == "minador" else (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    return img

def main(img_path):
    img, height, width, channels = load_image(img_path)
    outs = detect_objects(img)
    class_ids, confidences, boxes = get_boxes(outs, height, width)
    img = draw_labels(img, class_ids, confidences, boxes)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("../data_yolo/validation/test_image1.jpg")