import cv2
import numpy as np

# 画像サイズ
img_width = 600
img_height = 200
image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

# 梁（青）
beam_y = 150
cv2.line(image, (50, beam_y), (550, beam_y), (255, 0, 0), 4)

# 支点の共通設定
triangle_height = 30
support_color = (0, 0, 255)  # 赤（左右共通）

# 左支点（三角形・赤）
left_support_base = 50
triangle_left = np.array([
    [left_support_base, beam_y],
    [left_support_base - 15, beam_y + triangle_height],
    [left_support_base + 15, beam_y + triangle_height]
])
cv2.drawContours(image, [triangle_left], 0, support_color, -1)

# 右支点（三角形・赤）
right_support_base = 550
triangle_right = np.array([
    [right_support_base, beam_y],
    [right_support_base - 15, beam_y + triangle_height],
    [right_support_base + 15, beam_y + triangle_height]
])
cv2.drawContours(image, [triangle_right], 0, support_color, -1)

# 荷重（矢印・緑）
arrow_x = (50 + 550) // 2
arrow_y_start = beam_y - 50
arrow_y_end = beam_y - 4
cv2.arrowedLine(image, (arrow_x, arrow_y_start), (arrow_x, arrow_y_end), (0, 255, 0), 4, tipLength=0.4)

# 保存＆表示
cv2.imwrite("beam_diagram_red_supports.png", image)
cv2.imshow("Red Supports Beam", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


