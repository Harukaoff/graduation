from ultralytics import YOLO

def main():
    # 学習用 OBB モデルロード
    model = YOLO("yolov8n-obb.pt")

    # データ増強込みで学習
    model.train(
        data="dataset/data.yaml",
        epochs=50,
        imgsz=960,
        batch=16,
        workers=2,
        # データ増強の設定
        augment=True,
        degrees=10.0,     # 回転 ±10°
        scale=0.5,        # 画像拡大縮小 (0.5〜1.5倍)
        shear=2.0,        # シアー変換
        perspective=0.0005, # 透視変換
        hsv_h=0.015,      # 色相変化
        hsv_s=0.7,        # 彩度変化
        hsv_v=0.4,        # 明度変化
        flipud=0.5,       # 上下反転
        fliplr=0.5        # 左右反転
    )

    # 評価
    model.val()

    # テスト推論（例: valid の画像フォルダ）
    results = model.predict("dataset/valid/images", save=True)

if __name__ == "__main__":
    main()
