from ultralytics import YOLO
import cv2
import time

MODEL = "yolov8n.pt"
DEVICE = "cpu"   # change to 0 for GPU later

model = YOLO(MODEL)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened. Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    results = model(frame, device=DEVICE, verbose=False)
    end = time.time()

    fps = 1 / (end - start)
    latency = (end - start) * 1000

    annotated = results[0].plot()

    cv2.putText(
        annotated,
        f"Device: {DEVICE} | FPS: {fps:.2f} | Latency: {latency:.1f} ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.imshow("YOLOv8n Webcam", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()