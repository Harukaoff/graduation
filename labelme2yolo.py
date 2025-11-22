import json
import os

def labelme_json_to_yolo_txt(json_file, save_dir, class_map):
    with open(json_file, 'r') as f:
        data = json.load(f)
    img_w = data['imageWidth']
    img_h = data['imageHeight']

    yolo_lines = []
    for shape in data['shapes']:
        label = shape['label']
        if label not in class_map:
            continue
        class_id = class_map[label]
        points = shape['points']

        # ポリゴンのバウンディングボックス取得
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        xmin = min(xs)
        xmax = max(xs)
        ymin = min(ys)
        ymax = max(ys)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")

    base_name = os.path.basename(json_file).replace('.json', '.txt')
    with open(os.path.join(save_dir, base_name), 'w') as f:
        f.write("\n".join(yolo_lines))

# 使い方例
class_map = {
    "pin": 0,
    "roller": 1,
    "beam": 2,
    "load": 3,
    "concentrated_load": 4,
    "fixed": 5,
    "moment": 6,
    "hinge": 7
}
json_folder = "path/to/labelme/jsons"
save_folder = "path/to/save/txts"

for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        labelme_json_to_yolo_txt(os.path.join(json_folder, filename), save_folder, class_map)
