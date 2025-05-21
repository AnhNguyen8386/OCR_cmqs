import os
import pandas as pd
from PIL import Image

IMG_DIR = ""

# File nhãn Excel
EXCEL_PATH = "labels.xlsx"

class_map = {
    "ho_ten": 0,
    "so": 1,
    "cap_bac": 2,
    "don_vi_cap": 3,
    "han_su_dung": 4
}

df = pd.read_excel(EXCEL_PATH)

LABEL_DIR = "cccd_dataset/labels/train"
os.makedirs(LABEL_DIR, exist_ok=True)

for filename, group in df.groupby("filename"):
    label_lines = []
    img_path = os.path.join(IMG_DIR, filename)

    if not os.path.exists(img_path):
        print(f"Ảnh không tồn tại: {img_path}")
        continue

    with Image.open(img_path) as img:
        w, h = img.size

    for _, row in group.iterrows():
        cls_id = class_map.get(row['field'])
        if cls_id is None:
            continue

        x_center = (row['x_min'] + row['x_max']) / 2 / w
        y_center = (row['y_min'] + row['y_max']) / 2 / h
        width = (row['x_max'] - row['x_min']) / w
        height = (row['y_max'] - row['y_min']) / h

        label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    label_file = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt"))
    with open(label_file, "w") as f:
        f.write("\n".join(label_lines))

print("Chuyển đổi hoàn tất!")
