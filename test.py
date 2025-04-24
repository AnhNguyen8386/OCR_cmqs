from ultralytics import YOLO
import cv2
import os

model_path = "best.pt"

new_images_dir = "img_test/"
output_dir = "output/"

os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)

image_files = [f for f in os.listdir(new_images_dir) if f.endswith(('.jpg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(new_images_dir, image_file)

    image = cv2.imread(image_path)

    results = model.predict(source=image_path, save=False, conf=0.5)

    for result in results:

        annotated_image = result.plot()

        output_path = os.path.join(output_dir, f"annotated_{image_file}")
        cv2.imwrite(output_path, annotated_image)
        print(f"Đã lưu kết quả tại: {output_path}")


    cv2.imshow("Result", annotated_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()