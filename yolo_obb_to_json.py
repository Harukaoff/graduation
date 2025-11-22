import os
import glob
import json
import math

# クラス定義
CLASSES = ["beam", "fixed", "hinge", "load", "moment_l", "moment_r", "pin", "roller", "UDL"]

def obb_to_angle(x1, y1, x2, y2):
    """1つ目と2つ目の点から角度を算出"""
    rad = math.atan2(y2 - y1, x2 - x1)
    deg = math.degrees(rad)
    return deg

def process_label_file(label_path):
    elements, supports, loads, moments, udls, nodes = [], [], [], [], [], []
    node_id = 0
    elem_id = 0

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))  # YOLO OBB: x1 y1 x2 y2 x3 y3 x4 y4
        cname = CLASSES[cls_id]

        # 左上から時計回り前提
        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        angle = obb_to_angle(x1, y1, x2, y2)

        if cname == "beam":
            n1 = {"id": node_id, "x": x1, "y": y1}
            n2 = {"id": node_id + 1, "x": x3, "y": y3}
            nodes.extend([n1, n2])
            elements.append({"id": elem_id, "type": cname, "nodes": [node_id, node_id + 1], "angle": angle})
            node_id += 2
            elem_id += 1

        elif cname in ["fixed", "pin", "roller", "hinge"]:
            n = {"id": node_id, "x": x1, "y": y1}
            nodes.append(n)
            supports.append({"id": elem_id, "type": cname, "node": node_id})
            node_id += 1
            elem_id += 1

        elif cname == "load":
            n = {"id": node_id, "x": x1, "y": y1}
            nodes.append(n)
            loads.append({"id": elem_id, "type": cname, "node": node_id, "angle": angle, "magnitude": 1.0})
            node_id += 1
            elem_id += 1

        elif cname in ["moment_l", "moment_r"]:
            n = {"id": node_id, "x": x1, "y": y1}
            nodes.append(n)
            sign = -1 if cname == "moment_l" else 1
            moments.append({"id": elem_id, "type": cname, "node": node_id, "magnitude": sign * 1.0})
            node_id += 1
            elem_id += 1

        elif cname == "UDL":
            n1 = {"id": node_id, "x": x1, "y": y1}
            n2 = {"id": node_id + 1, "x": x2, "y": y2}
            nodes.extend([n1, n2])
            udls.append({"id": elem_id, "type": cname, "nodes": [node_id, node_id + 1], "magnitude": 1.0})
            node_id += 2
            elem_id += 1

    return {
        "nodes": nodes,
        "elements": elements,
        "supports": supports,
        "loads": loads,
        "moments": moments,
        "udl": udls
    }

def convert_dataset(labels_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    for label_path in label_files:
        data = process_label_file(label_path)
        out_path = os.path.join(out_dir, os.path.basename(label_path).replace(".txt", ".json"))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Converted {label_path} -> {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_dir", required=True, help="YOLO OBB ラベルのあるフォルダ")
    parser.add_argument("--out_dir", required=True, help="変換後の JSON 出力先")
    args = parser.parse_args()

    convert_dataset(args.labels_dir, args.out_dir)
