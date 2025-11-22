import streamlit as st
import cv2
import numpy as np

st.title("鋭利な矢印付き 梁・支点・荷重 図")

# 画像サイズ・背景
img_width = 600
img_height = 200
image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

# 梁（青）
beam_y = 150
cv2.line(image, (50, beam_y), (550, beam_y), (255, 0, 0), 4)

# 支点（三角形・赤）
triangle_color = (0, 0, 255)
triangle_height = 30

left_support_base = 50
triangle_left = np.array([
    [left_support_base, beam_y],
    [left_support_base - 15, beam_y + triangle_height],
    [left_support_base + 15, beam_y + triangle_height]
])
cv2.drawContours(image, [triangle_left], 0, triangle_color, -1)

right_support_base = 550
triangle_right = np.array([
    [right_support_base, beam_y],
    [right_support_base - 15, beam_y + triangle_height],
    [right_support_base + 15, beam_y + triangle_height]
])
cv2.drawContours(image, [triangle_right], 0, triangle_color, -1)

# 荷重（矢印・緑）
arrow_color = (0, 255, 0)
arrow_x = (50 + 550) // 2
arrow_y_start = beam_y - 80
arrow_y_end = beam_y - 4
cv2.line(image, (arrow_x, arrow_y_start), (arrow_x, arrow_y_end), arrow_color, 4)

# 鋭利な矢じり（三角形）
arrow_tip = (arrow_x, arrow_y_end)
arrow_left = (arrow_x - 10, arrow_y_end - 30)
arrow_right = (arrow_x + 10, arrow_y_end - 30)
arrow_head = np.array([arrow_tip, arrow_left, arrow_right])
cv2.drawContours(image, [arrow_head], 0, arrow_color, -1)

# BGR -> RGB変換（StreamlitはRGB）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 画像表示
st.image(image_rgb, caption="鋭利な矢印付き 梁・支点・荷重 図", use_column_width=True)

# 保存用にバイト列化
_, buffer = cv2.imencode('.png', image)
png_bytes = buffer.tobytes()

# ダウンロードボタン
st.download_button(
    label="画像をPNGでダウンロード",
    data=png_bytes,
    file_name="beam_diagram_sharp_arrow.png",
    mime="image/png"
)
