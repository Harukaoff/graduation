import os
import random
import shutil
from pathlib import Path

# 元画像とラベルのパス
images_dir = Path("images")
labels_dir = Path("labels")

# 出力先（YOLO形式の構造）
output_base = Path("dataset")
train_images = output_base / "images" / "train"
val_images = output_base / "images" / "val"
train_labels = output_base / "labels" / "train"
val_labels = output_base / "labels" / "val"

# ディレクトリ作成
for d in [train_images, val_images, train_labels, val_labels]:
    d.mkdir(parents=True, exist_ok=True)

# 分割比率
val_ratio = 0.2

# すべての画像ファイルを取得（jpeg前提）
image_files = list(images_dir.glob("*.jpeg"))
random.shuffle(image_files)

val_count = int(len(image_files) * val_ratio)

for i, img_path in enumerate(image_files):
    label_path = labels_dir / (img_path.stem + ".txt")

    if not label_path.exists():
        print(f"Warning: ラベルが見つかりません: {label_path}")
        continue

    if i < val_count:
        shutil.copy(img_path, val_images / img_path.name)
        shutil.copy(label_path, val_labels / label_path.name)
    else:
        shutil.copy(img_path, train_images / img_path.name)
        shutil.copy(label_path, train_labels / label_path.name)

print("✅ train / val 分割完了！")
