import os
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from collections import defaultdict

SEGMENT_MODEL = "model_segment.pt"
CLASSIFY_MODEL = "model_classify.pth"
DETECT_MODEL = "detect_model.pt"


def load_vietocr():
    config = Cfg.load_config_from_name("vgg_transformer")
    config['device'] = 'cpu'
    config['predictor']['beamsearch'] = True
    return Predictor(config)

vietocr = load_vietocr()
yolo_segment = YOLO(SEGMENT_MODEL)
yolo_detect = YOLO(DETECT_MODEL)

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Lỗi ảnh: {image_path}")
    return Image.open(image_path).convert("RGB")

def detect_cmqs_region(image):
    results = yolo_segment.predict(image, conf=0.5)
    if not results or not results[0].boxes:
        return None
    x1, y1, x2, y2 = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
    return (x1, y1, x2, y2)

def crop_image(image, box):
    return image.crop(box)

def fix_rotation(image):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(CLASSIFY_MODEL, map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = torch.argmax(model(input_tensor), dim=1).item()

    if prediction == 1:
        return image.rotate(180)
    return image

def detect_fields(image):
    results = yolo_detect.predict(image, conf=0.4)[0]
    fields = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        fields.append({'label': label, 'box': (x1, y1, x2, y2)})
    return fields

def ocr_boxes(image, fields):
    for field in fields:
        x1, y1, x2, y2 = field['box']
        crop = image.crop((x1, y1, x2, y2))
        text = vietocr.predict(crop)
        field['text'] = text
        field['cx'] = (x1 + x2) / 2
        field['cy'] = (y1 + y2) / 2
    return fields

def merge_boxes_by_label_and_position(fields, x_thresh=40, y_thresh=25):
    grouped = defaultdict(list)
    for field in fields:
        grouped[field['label']].append(field)

    merged_result = {}
    for label, items in grouped.items():
        items.sort(key=lambda f: (f['cy'], f['cx']))
        lines = []
        current_line = [items[0]]

        for field in items[1:]:
            last = current_line[-1]
            same_line = abs(field['cy'] - last['cy']) < y_thresh
            close_enough = abs(field['cx'] - last['cx']) < x_thresh

            if same_line and close_enough:
                current_line.append(field)
            else:
                # Ghép thành dòng
                line_text = ' '.join(word['text'] for word in current_line)
                lines.append(line_text)
                current_line = [field]

        if current_line:
            lines.append(' '.join(word['text'] for word in current_line))

        # Gộp các dòng lại
        merged_result[label] = ' '.join(lines)

    return merged_result

def process_image(image_path):
    image = load_image(image_path)
    region_box = detect_cmqs_region(image)
    if region_box is None:
        print("Không phát hiện cmqs")
        return {}
    image = crop_image(image, region_box)
    image = fix_rotation(image)
    fields = detect_fields(image)
    fields = ocr_boxes(image, fields)
    result = merge_boxes_by_label_and_position(fields)
    return result
if __name__ == "__main__":
    image_path = "test_img.jpg"
    result = process_image(image_path)
    print("Thông tin :")
    for label, text in result.items():
        print(f"{label}: {text}")
