from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        device="cuda",
        verbose=False
    )

    annotated = results[0].plot()

    num_objects = len(results[0].boxes)

    cv2.putText(
        annotated,
        f"Objects: {num_objects}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("YOLO Object Counter", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()