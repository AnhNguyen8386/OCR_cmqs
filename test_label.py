import cv2
import os

img_path = "cmqs/2021-09-14T083316.455667_cmsq_front_442.jpg"
txt_path = "your_dataset/labels_txt/2021-09-14T083316.455667_cmsq_front_442.txt"

class_map = {
    0: "Số ID",
    1: "Họ tên",
    2: "Cấp bậc",
    3: "Đơn vị cấp",
    4: "Ngày cấp",
    5: "Hạn sử dụng"
}

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]

img = cv2.imread(img_path)
h, w = img.shape[:2]

with open(txt_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split()
    class_id = int(parts[0])
    x_center = float(parts[1]) * w
    y_center = float(parts[2]) * h
    box_w = float(parts[3]) * w
    box_h = float(parts[4]) * h

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    color = colors[class_id % len(colors)]
    label = class_map.get(class_id, f"class_{class_id}")

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imshow("Check Labels", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
