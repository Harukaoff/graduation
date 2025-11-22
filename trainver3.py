from ultralytics import YOLO

def main():
    # 学習用 OBB モデルロード
    model = YOLO("yolov8n-obb.pt")

    # データ増強込みで学習
    model.train(
        data="dataset/data.yaml",
        epochs=100,          
        imgsz=640,
        batch=16,
        workers=2,
    

        # データ増強パラメータ
        augment=True,
        degrees=7.5,        # 回転 ±7.5
        scale=0.6,           # 拡縮 (0.4〜1.6倍)
        shear=5.0,           # シアー変換
        perspective=0.001,   # 透視変換
        hsv_h=0.02,          # 色相
        hsv_s=0.9,           # 彩度 ← 強化
        hsv_v=0.6,           # 明度 ← 強化
        flipud=0.5,          # 上下反転
        fliplr=0.5           # 左右反転
    )

    # 評価
    model.val()

    # テスト推論（例: valid の画像フォルダ）
    results = model.predict("dataset/valid/images", save=True)

if __name__ == "__main__":
    main()
