import os
import random
import shutil

base_path = "your_dataset"
images_dir = os.path.join(base_path, "cmqs")
labels_dir = os.path.join(base_path, "labels_txt")

for split in ['train', 'val']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
random.shuffle(image_files)

train_ratio = 0.8
split_index = int(len(image_files) * train_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

def move_files(file_list, split):
    for img_file in file_list:
        label_file = img_file.rsplit(".", 1)[0] + ".txt"

        shutil.move(os.path.join(images_dir, img_file),
                    os.path.join(images_dir, split, img_file))

        shutil.move(os.path.join(labels_dir, label_file),
                    os.path.join(labels_dir, split, label_file))

move_files(train_files, "train")
move_files(val_files, "val")

print(f"Chia xong: {len(train_files)} train / {len(val_files)} val")
