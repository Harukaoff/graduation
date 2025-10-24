from PIL import Image, ImageDraw, ImageFont
import numpy as np
import streamlit as st
from ultralytics import YOLO
import os

# ======================
# テンプレート読み込み
# ======================
def load_template(path):
    try:
        return Image.open(path).convert("RGBA")
    except:
        st.warning(f"テンプレートが読み込めません: {path}")
        return None

templates = {
    "pin": load_template("templates/pin.png"),
    "roller": load_template("templates/roller.png"),
    "fixed": load_template("templates/fixed.png"),
    "beam": load_template("templates/beam.png"),
    "load": load_template("templates/load.png"),
    "moment l": load_template("templates/momentL.png"),
    "moment r": load_template("templates/momentR.png"),
    "udl": load_template("templates/UDL.png"),
}

# ======================
# テンプレート貼り付け関数
# ======================
def overlay_template(base_img, template_img, center, angle, scale=1.0):
    # スケーリング
    w, h = template_img.size
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w <= 0 or new_h <= 0:
        return base_img

    template_resized = template_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 回転（Pillowは反時計回り）
    rotated = template_resized.rotate(angle, expand=True)

    # 中心座標から貼り付け位置を計算
    x, y = center
    paste_x = int(x - rotated.width / 2)
    paste_y = int(y - rotated.height / 2)

    base_img.paste(rotated, (paste_x, paste_y), mask=rotated)
    return base_img

# ======================
# Streamlit アプリ
# ======================
def run_app():
    st.title("手書き画像認識　自動清書アプリ（PIL版）")
    st.write("YOLOv8 OBB で要素を認識し、テンプレートに置換して白紙上に清書します。")

    model = YOLO("runs/obb/train28/weights/best.pt")

    conf_th = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img_pil = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img_pil)

        results = model(img_np, conf=conf_th, imgsz=640)[0]

        # 白紙キャンバス作成
        canvas = Image.new("RGBA", img_pil.size, (255, 255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        if hasattr(results, "obb") and results.obb is not None:
            for box in results.obb:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                if conf < conf_th:
                    continue

                name = results.names[cls_id]
                if name not in templates or templates[name] is None:
                    continue

                x, y, w, h, angle = box.xywhr.cpu().numpy()[0]
                angle_deg = -float(angle * 180 / np.pi)
                template = templates[name]
                scale = max(w / template.width, 0.1)

                canvas = overlay_template(canvas, template, (x, y), angle_deg, scale)
                draw.text((x, y - 10), f"{name} {conf:.2f}", fill=(255, 0, 0))

        # 結果を表示
        st.image(canvas, caption="清書結果（白紙キャンバス上）", use_container_width=True)

        os.makedirs("output", exist_ok=True)
        output_path = "output/output_overlay.png"
        canvas.save(output_path)
        with open(output_path, "rb") as f:
            st.download_button("結果をダウンロード", f, file_name="output_overlay.png")

if __name__ == "__main__":
    run_app()

