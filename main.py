import cv2
import numpy as np
import sys
from tensorflow.keras.models import load_model

model = load_model("models/digit_model.h5")

print("Available cameras:")

if sys.platform == "darwin":  # macOS
    backend = cv2.CAP_AVFOUNDATION
elif sys.platform == "win32":  # Windows
    backend = cv2.CAP_DSHOW
else:  # Linux и всё остальное
    backend = 0

for i in range(5):
    cap_test = cv2.VideoCapture(i, backend)
    if not cap_test.isOpened():
        cap_test.release()
        continue

    # Попробуем прочитать один кадр, но без шума в лог
    ret, _ = cap_test.read()
    if ret:
        print(f"[{i}] Камера доступна")
    cap_test.release()

cam_index = input("Enter camera index (default is 0): ")
try:
    cam_index = int(cam_index)
except ValueError:
    cam_index = 0

cap = cv2.VideoCapture(cam_index)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame is None:
        continue

    frame_height, frame_width = frame.shape[:2]
    box_size = 300
    x1 = frame_width // 2 - box_size // 2
    y1 = frame_height // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)
    cv2.imshow("threshold", thresh)

    img = cv2.resize(thresh, (28, 28))
    img = img / 255.0
    img = img.reshape(1, 28, 28)

    pred = model.predict(img)
    digit = np.argmax(pred)

    cv2.putText(frame, f"Digit: {digit}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("digit_recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()