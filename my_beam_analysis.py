import cv2
import numpy as np

# 画像読み込み & リサイズ
img = cv2.imread("IMG_A159C82A-2329-40E9-AE58-B6B0B290AE78.jpeg")
img = cv2.resize(img, (800, 600))

# グレースケール & 二値化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

# 輪郭抽出
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    x, y, w, h = cv2.boundingRect(approx)

    # 支点（三角形）検出
    if len(approx) == 3:
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
        print("Triangle (支点) detected at:", x + w // 2)

    # 荷重（矢印） → 縦長の図形として検出
    elif h > w * 1.5:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print("Vertical shape (矢印？) at:", x + w // 2)

    # 梁（直線） → 横長なものとして検出
    elif w > h * 3:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print("Beam (梁) at:", x, y)

# 仮に検出されたx座標（px単位）
left_support_x = 200
right_support_x = 600
load_x = 400
load_value = 10  # N

# 長さ（単位は任意スケールに変換可能）
L = right_support_x - left_support_x
a = load_x - left_support_x
b = L - a

# 反力計算
RA = load_value * b / L
RB = load_value * a / L

print(f"左端支点反力 RA = {RA:.2f} N")
print(f"右端支点反力 RB = {RB:.2f} N")
