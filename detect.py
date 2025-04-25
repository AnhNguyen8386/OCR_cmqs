import os
import cv2
from ultralytics import YOLO
import numpy as np

def extract_bounding_box_region(image, mask):

    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    cropped_region = image[y:y + h, x:x + w]

    return cropped_region


def segment_and_crop_cmqs(image_path, model, class_id=0):

    image = cv2.imread(image_path)

    results = model(image)[0]

    if len(results.masks) == 0:
        print(f"Không phát hiện CMQS trong {image_path}.")
        return None

    cropped_images = []
    for i, mask in enumerate(results.masks.data):
        cls = int(results.boxes.cls[i].item())
        if cls != class_id:
            continue

        mask_np = mask.cpu().numpy()

        mask_resized = cv2.resize(mask_np, (image.shape[1], image.shape[0]))

        cropped_region = extract_bounding_box_region(image, mask_resized)

        if cropped_region is not None:
            cropped_images.append(cropped_region)

    return cropped_images


def process_folder(input_folder, output_folder, model_path='best.pt', class_id=0):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = YOLO(model_path)

    for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        print(f"Đang xử lý ảnh: {filename}")

        cropped_images = segment_and_crop_cmqs(image_path, model, class_id)

        if cropped_images is None:
            continue

        for i, cropped_image in enumerate(cropped_images):
            save_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_cropped_{i}.png")
            cv2.imwrite(save_path, cropped_image)
            print(f"Đã lưu: {save_path}")


input_folder = 'img_test'
output_folder = 'cropped_img'

process_folder(input_folder, output_folder, model_path='best.pt', class_id=0)
