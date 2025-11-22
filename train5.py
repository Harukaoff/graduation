from ultralytics import YOLO

def main():
    # モデル読み込み（OBB対応）
    model = YOLO("yolov8n-obb.pt")

    # 学習設定
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=2,
        augment=True,     # データ増強を有効化

        # ========= データ増強パラメータ =========
        flipud=0.0,       # 上下反転なし
        fliplr=0.0,       # 左右反転なし
        degrees=15.0,     # ±15°回転（自然な傾き）
        scale=0.5,        # 拡縮（0.5〜1.5倍）
        shear=0.0,        # シアー（傾き）なし
        perspective=0.0,  # 透視変換なし

        # === 色変換（照明や色むら対策） ===
        hsv_h=0.02,       # 色相シフト（少なめ）
        hsv_s=0.8,        # 彩度シフト（大きめ）
        hsv_v=0.6,        # 明度シフト（やや強め）
        # =====================================
    )

    # 検証
    model.val()

    # 予測（確認用）
    model.predict("dataset/valid/images", save=True)

if __name__ == "__main__":
    main()
