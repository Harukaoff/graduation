from ultralytics import YOLO

def main():
    # 学習用 OBB モデルロード
    model = YOLO("yolov8n-obb.pt")

    # 学習開始
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
    )

    # 評価
    model.val()

    # テスト推論（例: valid の1枚）
    results = model.predict("dataset/valid/images", save=True)

if __name__ == "__main__":
    main()
