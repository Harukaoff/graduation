import cv2

# ① 画像の読み込み（※パスは実際のファイルに合わせて）
image_path = "C:/Users/morim/Downloads/IMG_5841.JPG"
img = cv2.imread(image_path)

# ② 読み込みチェック
if img is None:
    print("画像が読み込めません。パスが正しいか確認して！")
    exit()

# ③ グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ④ 2値化（しきい値は必要に応じて調整）
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# ⑤ 輪郭検出
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ⑥ 輪郭を元の画像に描画（緑）
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# ⑦ 結果をウィンドウ表示
cv2.imshow("Detected Contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

