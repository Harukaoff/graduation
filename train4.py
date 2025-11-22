from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-obb.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
    )

    model.val()
    model.predict("dataset/valid/images", save=True)

if __name__ == "__main__":
    main()
