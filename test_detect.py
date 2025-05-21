from ultralytics import YOLO
import cv2
import os

model = YOLO("detect_model.pt")

folder_path = "cropped_img"
crop_folder = "crop_fields"
os.makedirs(crop_folder, exist_ok=True)

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    results = model(image_path)

    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Không đọc được ảnh {image_path}")
        continue

    for i, box in enumerate(results[0].boxes):
        label = results[0].names[int(box.cls)]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_img.shape[1], x2)
        y2 = min(original_img.shape[0], y2)

        crop_img = original_img[y1:y2, x1:x2]

        crop_filename = f"{os.path.splitext(image_file)[0]}_{label}_{i}.jpg"
        crop_path = os.path.join(crop_folder, crop_filename)

        cv2.imwrite(crop_path, crop_img)
        print(f"Crop vùng '{label}' từ {image_file} lưu vào: {crop_path}")
