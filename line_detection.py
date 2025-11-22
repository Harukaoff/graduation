import cv2
import numpy as np

# 画像のサイズ
img_width = 600
img_height = 200
image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

# 梁の位置
beam_y = 150
cv2.line(image, (50, beam_y), (550, beam_y), (0, 0, 0), 6)

# 左支点（三角形）
left_support_base = 50
triangle_height = 30
triangle_left = np.array([
    [left_support_base, beam_y],
    [left_support_base - 15, beam_y + triangle_height],
    [left_support_base + 15, beam_y + triangle_height]
])
cv2.drawContours(image, [triangle_left], 0, (0, 0, 0), -1)

# 右支点（三角形）
right_support_base = 550
triangle_right = np.array([
    [right_support_base, beam_y],
    [right_support_base - 15, beam_y + triangle_height],
    [right_support_base + 15, beam_y + triangle_height]
])
cv2.drawContours(image, [triangle_right], 0, (0, 0, 0), -1)

# 荷重（中央の下向き矢印）
arrow_x = (50 + 550) // 2
arrow_y_start = beam_y - 50
arrow_y_end = beam_y
cv2.arrowedLine(image, (arrow_x, arrow_y_start), (arrow_x, arrow_y_end), (0, 0, 0), 4, tipLength=0.2)

# 保存＆表示
cv2.imwrite("beam_diagram_fixed.png", image)
cv2.imshow("Fixed Beam", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 仮の入力データ（例）
beam_data = {
    "beam_length": 10.0,
    "supports": [
        {"type": "pin", "position": 0.0},
        {"type": "roller", "position": 10.0}
    ],
    "loads": [
        {"type": "point", "position": 4.0, "magnitude": 10.0}
    ]
}

# 反力計算関数
def calculate_reactions(beam_length, supports, loads):
    A = supports[0]['position']
    B = supports[1]['position']
    L = B - A

    RA = 0
    RB = 0

    # 荷重ごとにモーメントと力の合計を取る
    for load in loads:
        if load['type'] == 'point':
            P = load['magnitude']
            x = load['position'] - A  # 左端からの距離

            # モーメントの釣り合いからRB
            RB += (P * x) / L

    # 垂直方向の釣り合いからRA
    total_load = sum([load['magnitude'] for load in loads])
    RA = total_load - RB

    return RA, RB

# 実行
RA, RB = calculate_reactions(
    beam_length=beam_data["beam_length"],
    supports=beam_data["supports"],
    loads=beam_data["loads"]
)

print(f"左端支点反力 RA = {RA:.2f} N")
print(f"右端支点反力 RB = {RB:.2f} N")


