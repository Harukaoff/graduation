import os
import json
import glob
from PIL import Image

# クラス名一覧（ラベル名と順番は YOLO のクラスID に対応）
CLASS_NAMES = ["pin", "roller", "fixed", "arrow", "hinge", "moment", "beam", "load"]

# パス設定
IMAGE_DIR = "dataset/images"
ANNOTATION_DIR = "dataset/annotations"
OUTPUT_LABEL_DIR = "dataset/labels"

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

for json_path in glob.glob(os.path.join(ANNOTATION_DIR, "*.json")):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_filename = data["imagePath"]
    image_path = os.path.join(IMAGE_DIR, image_filename)

    if not os.path.exists(image_path):
        print(f"[画像が見つからない] {image_filename}")
        continue

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    base_filename = os.path.splitext(os.path.basename(json_path))[0]
    output_txt_path = os.path.join(OUTPUT_LABEL_DIR, base_filename + ".txt")

    with open(output_txt_path, "w", encoding="utf-8") as out_file:
        for shape in data["shapes"]:
            label = shape["label"].strip().lower()
            if label not in CLASS_NAMES:
                print(f"[未定義ラベル] {label} in {json_path}")
                continue
            class_id = CLASS_NAMES.index(label)

            points = shape["points"]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # YOLO形式へ変換
            cx = (x_min + x_max) / 2 / img_width
            cy = (y_min + y_max) / 2 / img_height
            w = (x_max - x_min) / img_width
            h = (y_max - y_min) / img_height

            out_file.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
