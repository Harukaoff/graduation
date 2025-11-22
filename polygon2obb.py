import os
import numpy as np

def poly_to_obb(coords):
    """
    polygon (x1,y1...x4,y4) → (cx, cy, w, h, angle[deg])
    頂点は左上から時計回りで来ている前提
    """
    pts = np.array(coords, dtype=np.float32).reshape(4, 2)

    # 中心
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    # 幅: p1→p2, 高さ: p2→p3
    w = np.linalg.norm(pts[1] - pts[0])
    h = np.linalg.norm(pts[2] - pts[1])

    # 角度 (p1→p2 の傾き)
    dx, dy = pts[1] - pts[0]
    angle = np.degrees(np.arctan2(dy, dx))

    return cx, cy, w, h, angle


def convert_labels(in_dir, out_dir, img_width, img_height):
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(in_dir):
        if not file.endswith(".txt"):
            continue

        in_path = os.path.join(in_dir, file)
        out_path = os.path.join(out_dir, file)

        with open(in_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            cx, cy, w, h, angle = poly_to_obb(coords)

            # 正規化
            cx /= img_width
            cy /= img_height
            w /= img_width
            h /= img_height

            new_line = f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {angle:.6f}\n"
            new_lines.append(new_line)

        with open(out_path, "w") as f:
            f.writelines(new_lines)


if __name__ == "__main__":
    # 画像サイズ（固定なら指定。可変ならOpenCVで読み込むほうが正確）
    IMG_W, IMG_H = 640, 640  

    for split in ["train", "valid"]:
        in_dir = f"dataset/{split}/labels"
        out_dir = f"dataset/{split}/labels_obb"
        convert_labels(in_dir, out_dir, IMG_W, IMG_H)

    print("✅ Polygon → OBB 変換完了！")
