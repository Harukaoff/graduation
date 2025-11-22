from streamlit_drawable_canvas import st_canvas
import streamlit as st
from PIL import Image

st.title("構造図のインタラクティブ編集")

# アップロードされた画像の最大サイズ
MAX_WIDTH = 800
MAX_HEIGHT = 600

uploaded_file = st.file_uploader("構造図画像をアップロード", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # 画像を最大サイズにリサイズ（アスペクト比維持）
    image.thumbnail((MAX_WIDTH, MAX_HEIGHT))
    
    # canvas表示（画像サイズに合わせる）
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="point",  # クリックした位置の座標を取得
        key="canvas",
    )
    
    # クリックで追加した点の座標を表示
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            x, y = obj["left"], obj["top"]
            st.write(f"追加された点: ({x:.1f}, {y:.1f})")
