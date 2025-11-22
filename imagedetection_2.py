import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load and preprocess the image ---
image_path = "C:/Users/morim/Downloads/IMG_5841.JPG"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# --- Step 2: Detect beam (assumed to be the longest horizontal line) ---
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=10)
beam_line = None
max_length = 0
for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.hypot(x2 - x1, y2 - y1)
    if length > max_length and abs(y1 - y2) < 10:  # horizontal line check
        max_length = length
        beam_line = (x1, y1, x2, y2)

# Draw detected beam
if beam_line:
    x1, y1, x2, y2 = beam_line
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    beam_length = 600  # assume 600 mm for scaling
    load_position = (x1 + x2) // 2  # assume load is at center
else:
    raise ValueError("梁（beam）が検出できませんでした")

# --- Step 3: Calculate reaction forces (仮：中央集中荷重、10N) ---
P = 10  # Load (N)
a = beam_length / 2
b = beam_length / 2
RA = P * b / (a + b)
RB = P * a / (a + b)

print(f"左端支点反力 RA = {RA:.2f} N")
print(f"右端支点反力 RB = {RB:.2f} N")

# --- Step 4: Draw Shear Force Diagram ---
x = [0, a, a, beam_length]
V = [RA, RA, -RB, -RB]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, V, drawstyle='steps-post')
plt.title("Shear Force Diagram")
plt.xlabel("x [mm]")
plt.ylabel("V [N]")
plt.grid(True)

# --- Step 5: Draw Bending Moment Diagram ---
xm = np.linspace(0, beam_length, 100)
M = np.piecewise(xm, [xm <= a, xm > a],
                 [lambda x: RA * x, lambda x: RA * x - P * (x - a)])

plt.subplot(1, 2, 2)
plt.plot(xm, M)
plt.title("Bending Moment Diagram")
plt.xlabel("x [mm]")
plt.ylabel("M [Nmm]")
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Optional: Show the processed image ---
cv2.imshow("Detected Beam", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
