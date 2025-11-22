import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# ======================
# テンプレート読み込み
# ======================
def load_template(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if temp is None:
        st.warning(f"テンプレートが読み込めません: {path}")
        return None
    # アルファチャンネルがない場合は追加
    if temp.shape[2] == 3:
        b, g, r = cv2.split(temp)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        temp = cv2.merge([b, g, r, alpha])
    return temp

templates = {
    "pin": load_template("templates/pin.jpg"),
    "roller": load_template("templates/roller.jpg"),
    "fixed": load_template("templates/fixed.jpg"),
    "beam": load_template("templates/beam.jpg"),
    "load": load_template("templates/load.jpg"),
    "moment_l": load_template("templates/momentL.jpg"),
    "moment_r": load_template("templates/momentR.jpg"),
    "udl": load_template("templates/UDL.jpg"),
}

# ======================
# テンプレート貼り付け関数
# ======================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    """テンプレートを回転・スケーリングして白地に貼り付ける"""
    h, w = template_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return base_img

    # リサイズ
    template_resized = cv2.resize(template_img, (new_w, new_h))

    # 回転
    M = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        template_resized, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # 貼り付け座標計算
    x, y = int(center[0]), int(center[1])
    x1, y1 = max(x - new_w // 2, 0), max(y - new_h // 2, 0)
    x2, y2 = min(x1 + new_w, base_img.shape[1]), min(y1 + new_h, base_img.shape[0])
    roi = base_img[y1:y2, x1:x2]

    # サイズ調整
    rot_crop = rotated[0:(y2 - y1), 0:(x2 - x1)]

    # アルファブレンド
    alpha = rot_crop[:, :, 3] / 255.0
    for c in range(3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * rot_crop[:, :, c]

    base_img[y1:y2, x1:x2] = roi
    return base_img

# ======================
# Streamlit アプリ
# ======================
def run_app():
    st.title("構造図 自動清書アプリ（白地キャンバス版）")
    st.write("YOLOv8 OBBで要素を認識し、白背景にテンプレートを再配置します。")

    # モデルロード
    model = YOLO("runs/obb/train28/weights/best.pt")

    # 信頼度スライダー
    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)

    # 画像アップロード
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape

        # 真っ白キャンバス（RGBA）
        canvas = np.ones((h, w, 4), dtype=np.uint8) * 255

        # 推論
        results = model(img, conf=conf_th, imgsz=640)[0]

        # 検出されたOBBを処理
        if hasattr(results, "obb") and results.obb is not None:
            for box in results.obb:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                if conf < conf_th:
                    continue

                name = results.names[cls_id]
                if name not in templates or templates[name] is None:
                    continue

                # xywhr形式（中心x, 中心y, 幅, 高さ, 角度[rad]）
                x, y, w_box, h_box, angle = box.xywhr.cpu().numpy()[0]
                angle_deg = float(angle * 180 / np.pi)

                template = templates[name]
                scale = max(w_box / template.shape[1], 0.1)

                # テンプレートを白地キャンバスに配置
                canvas = overlay_template(canvas, template, (x, y), angle_deg, scale)

                # オプション: 信頼度ラベル
                cv2.putText(
                    canvas,
                    f"{name} {conf:.2f}",
                    (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        # 結果表示
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
        st.image(img_rgb, caption="清書結果（白地キャンバス）", use_container_width=True)

        # ダウンロードボタン
        result_pil = Image.fromarray(img_rgb)
        os.makedirs("output", exist_ok=True)
        save_path = "output/clean_canvas.png"
        result_pil.save(save_path)
        with open(save_path, "rb") as f:
            st.download_button("清書画像をダウンロード", f, file_name="clean_canvas.png")

if __name__ == "__main__":
    run_app()
