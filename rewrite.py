import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === 設定（テンプレート画像の読み込み） ===
template_pin = cv2.imread('C:/Users/morim/Downloads/卒業研究/templates/pin.png', cv2.IMREAD_UNCHANGED)
template_pinroller = cv2.imread('C:/Users/morim/Downloads/卒業研究/templates/roller.png', cv2.IMREAD_UNCHANGED)
template_fixed = cv2.imread('C:/Users/morim/Downloads/卒業研究/templates/fixed.png', cv2.IMREAD_UNCHANGED)

# === 画像サイズなどの共通設定 ===
canvas_w, canvas_h = 1000, 600
canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255  # 白背景

# === 座標変換関数 ===
def paste_template(bg, template, center_x, center_y, scale=1.0):
    h, w = template.shape[:2]
    resized = cv2.resize(template, (int(w * scale), int(h * scale)))
    th, tw = resized.shape[:2]

    x1 = int(center_x - tw // 2)
    y1 = int(center_y - th // 2)
    x2 = x1 + tw
    y2 = y1 + th

    if resized.shape[2] == 4:
        alpha = resized[:, :, 3] / 255.0
        for c in range(3):
            bg[y1:y2, x1:x2, c] = (1 - alpha) * bg[y1:y2, x1:x2, c] + alpha * resized[:, :, c]
    else:
        bg[y1:y2, x1:x2] = resized[:, :, :3]
    return bg

# === 梁の描画 ===
def draw_beam(canvas, start, end, thickness=8):
    return cv2.line(canvas, start, end, (0, 0, 0), thickness)

# === 荷重の描画（超鋭利な矢印） ===
def draw_sharp_arrow(canvas, base, direction='down', length=80, color=(0, 0, 0), thickness=3):
    x, y = base
    if direction == 'down':
        tip = (x, y + length)
    elif direction == 'up':
        tip = (x, y - length)
    elif direction == 'left':
        tip = (x - length, y)
    else:
        tip = (x + length, y)
    
    # 鋭利な矢印
    canvas = cv2.arrowedLine(canvas, base, tip, color, thickness, tipLength=0.5)
    return canvas

# === 再構成（仮配置） ===
beam_start = (200, 300)
beam_end = (800, 300)
canvas = draw_beam(canvas, beam_start, beam_end)

load_point = (500, 300)
canvas = draw_sharp_arrow(canvas, load_point, direction='down')

canvas = paste_template(canvas, template_pin, center_x=200, center_y=300)
canvas = paste_template(canvas, template_pinroller, center_x=800, center_y=300)
canvas = paste_template(canvas, template_fixed, center_x=200, center_y=500)

# === 表示 ===
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


