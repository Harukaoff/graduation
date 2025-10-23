import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# -------------------------------
# Streamlit App
# -------------------------------
def run_app():
    st.title("構造解析自動化アプリ（YOLO + FEM）")
    st.write("画像から支点・梁・荷重を検出して、構造解析を行います。")

    # モデルパス
    model_path = "runs/obb/train28/weights/best.pt"
    st.sidebar.header("設定")
    conf_thres = st.sidebar.slider("検出信頼度しきい値", 0.0, 1.0, 0.25)

    # モデル読み込み
    st.write("YOLOモデル読み込み中...")
    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=model_path,
        source="github",
        force_reload=True
    )
    st.success("モデル読み込み完了！")

    # 画像アップロード
    uploaded_file = st.file_uploader("構造画像をアップロード", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="入力画像", use_container_width=True)

        # 検出
        st.write("構造要素を検出中...")
        results = model(image, size=640)
        df = results.pandas().xyxy[0]
        df = df[df["confidence"] >= conf_thres]

        if df.empty:
            st.error("要素が検出されませんでした。")
            return

        st.dataframe(df[["name", "xmin", "ymin", "xmax", "ymax", "confidence"]])

        # 清書画像を表示
        img_bytes = io.BytesIO()
        results.render()
        Image.fromarray(results.ims[0]).save(img_bytes, format="PNG")
        st.image(Image.open(img_bytes), caption="検出結果", use_container_width=True)

        # 要素情報をelem_infoに変換
        elem_info = []
        for _, row in df.iterrows():
            name = row["name"]
            cx = (row["xmin"] + row["xmax"]) / 2
            cy = (row["ymin"] + row["ymax"]) / 2
            elem_info.append({"type": name, "x": cx, "y": cy})

        st.write("抽出された要素情報：", elem_info)

        # -------------------------------
        # 構造解析フェーズ
        # -------------------------------
        n_d, e_l = generate_structure(elem_info)
        st.write("節点リスト:", n_d)
        st.write("要素リスト:", e_l)

        # 解析
        disp = d_r(n_d, e_l)
        stress = member_stress(n_d, e_l, disp)

        st.subheader("解析結果")
        st.write("節点変位:", disp)
        st.write("部材応力:", stress)

        # 応力図描画
        plot_stress_diagram(n_d, e_l, stress)
        st.pyplot()


# -------------------------------
# 構造解析用関数群
# -------------------------------
def generate_structure(elem_info):
    """YOLO検出結果から節点リスト・要素リストを自動生成"""
    nodes = []
    beams = []

    for e in elem_info:
        if e["type"] in ["pin", "roller", "fixed"]:
            nodes.append([e["x"], e["y"]])
        elif e["type"] == "beam":
            beams.append(e)

    n_d = np.array(nodes)
    e_l = []
    for i in range(len(beams)):
        if len(n_d) >= 2:
            e_l.append([0, 1])  # 仮に最初の2点を接続（シンプルな例）
    return n_d, e_l


def d_r(n_d, e_l):
    """簡易変位計算"""
    disp = np.zeros_like(n_d)
    disp[:, 1] = np.linspace(0, -0.1, len(n_d))
    return disp


def member_stress(n_d, e_l, disp):
    """簡易応力計算"""
    stress = []
    E = 2.1e8
    A = 0.01
    for e in e_l:
        n1, n2 = e
        L0 = np.linalg.norm(n_d[n2] - n_d[n1])
        L = np.linalg.norm((n_d[n2] + disp[n2]) - (n_d[n1] + disp[n1]))
        strain = (L - L0) / L0
        stress.append(E * strain)
    return np.array(stress)


def plot_stress_diagram(n_d, e_l, stress):
    plt.figure()
    for i, e in enumerate(e_l):
        n1, n2 = e
        x = [n_d[n1, 0], n_d[n2, 0]]
        y = [n_d[n1, 1], n_d[n2, 1]]
        plt.plot(x, y, "k-", lw=2)
        plt.text(np.mean(x), np.mean(y), f"{stress[i]:.2e}", color="red")
    plt.title("応力図")
    plt.axis("equal")
    plt.gca().invert_yaxis()


# -------------------------------
# 実行
# -------------------------------
if __name__ == "__main__":
    run_app()
