import cv2, numpy as np

cap = cv2.VideoCapture('d:/SIH/uploads/sample_cctv_night.mp4')
ret, frame = cap.read()
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print("Min gray:", np.min(gray), "Max gray:", np.max(gray), "Mean gray:", np.mean(gray))

# Let's test simple OTSU or adaptive threshold
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enh = clahe.apply(gray)
print("Enhanced min:", np.min(enh), "max:", np.max(enh), "mean:", np.mean(enh))

# Find contours with Otsu on ground region
ground = enh[int(frame.shape[0]*0.45):, :]
_, th = cv2.threshold(ground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Found {len(cnts)} contours on ground with Otsu:")
for c in cnts[:5]:
    x, y, bw, bh = cv2.boundingRect(c)
    print(f"  bbox: [{x}, {y}, {bw}, {bh}], area: {cv2.contourArea(c)}")
