import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_support_reactions(length, load_position, load_magnitude):
    a = load_position
    b = length - load_position
    RA = (load_magnitude * b) / length
    RB = (load_magnitude * a) / length
    return RA, RB

def calculate_shear_force_and_moment(length, load_position, load_magnitude, RA):
    x_vals = np.linspace(0, length, 500)
    shear_force = []
    moment = []
    for x in x_vals:
        if x < load_position:
            V = RA
            M = RA * x
        else:
            V = RA - load_magnitude
            M = RA * x - load_magnitude * (x - load_position)
        shear_force.append(V)
        moment.append(-M)  # 符号反転で正のモーメントを上向きに
    return x_vals, shear_force, moment

def plot_diagrams(x_vals, shear_force, moment):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(x_vals, shear_force, label='Shear Force', color='blue')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('Shear Force Diagram')
    plt.ylabel('Shear Force [N]')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x_vals, moment, label='Bending Moment', color='red')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title('Bending Moment Diagram')
    plt.xlabel('Position [m]')
    plt.ylabel('Moment [N·m]')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    image_path = "C:/Users/morim/Downloads/IMG_A159C82A-2329-40E9-AE58-B6B0B290AE78.jpeg"
    img = cv2.imread(image_path)
    if img is None:
        print("画像を読み込めませんでした。パスを確認してください。")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("構造線が検出できませんでした。")
        return

    # 仮の梁長さと荷重位置（現状は手動設定）
    beam_length = 10  # 単位は m
    load_position = 5  # 中央に荷重（仮）
    load_magnitude = 10  # N（仮）

    RA, RB = calculate_support_reactions(beam_length, load_position, load_magnitude)
    print(f"左端支点反力 RA = {RA:.2f} N")
    print(f"右端支点反力 RB = {RB:.2f} N")

    x_vals, shear_force, moment = calculate_shear_force_and_moment(beam_length, load_position, load_magnitude, RA)
    plot_diagrams(x_vals, shear_force, moment)

if __name__ == "__main__":
    main()
    