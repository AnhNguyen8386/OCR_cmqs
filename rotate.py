import os
import cv2
import shutil
import random


src_dir = 'your_dataset/cmqs'

dst_base = 'dataset'
train_ratio = 0.8


for split in ['train', 'val']:
    for label in ['0', '1']:
        os.makedirs(os.path.join(dst_base, split, label), exist_ok=True)


image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(image_files)


split_index = int(len(image_files) * train_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def process_and_save(files, split):
    for file in files:
        img_path = os.path.join(src_dir, file)
        img = cv2.imread(img_path)


        dst_0 = os.path.join(dst_base, split, '0', file)
        cv2.imwrite(dst_0, img)


        rotated = cv2.rotate(img, cv2.ROTATE_180)
        dst_1 = os.path.join(dst_base, split, '1', 'rotated_' + file)
        cv2.imwrite(dst_1, rotated)


process_and_save(train_files, 'train')
process_and_save(val_files, 'val')

print("Đã tạo dataset")
