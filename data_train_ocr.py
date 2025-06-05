import os
import cv2
from ultralytics import YOLO

input_folder = "cropped_img"
output_folder = "output"
model_path = "detect_model.pt"
model = YOLO(model_path)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print("Ảnh lỗi:", filename)
            continue

        results = model.predict(image_path, verbose=False)
        boxes = results[0].boxes
        names = results[0].names

        box_list = []
        for i in range(len(boxes.data)):
            box = boxes.data[i]
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            class_id = int(box[5])
            label = names[class_id]
            box_list.append((x1, y1, x2, y2, label))

        box_list.sort(key=lambda b: (b[4], b[0]))

        label_counter = {}

        for box in box_list:
            x1, y1, x2, y2, label = box

            if label not in label_counter:
                label_counter[label] = 1
            else:
                label_counter[label] += 1
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            name_no_ext = os.path.splitext(filename)[0]
            save_name = f"{name_no_ext}_{label}_{label_counter[label]}.png"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, cropped)

print("crop xong")
