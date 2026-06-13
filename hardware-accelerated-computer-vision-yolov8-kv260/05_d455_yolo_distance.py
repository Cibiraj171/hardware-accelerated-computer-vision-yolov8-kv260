from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import time

model = YOLO("yolov8n.pt")

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())

        start = time.time()
        results = model.predict(frame, device="cuda", verbose=False)
        end = time.time()

        annotated = results[0].plot()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            distance = depth_frame.get_distance(cx, cy)

            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                f"{label}: {distance:.2f} m",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        fps = 1 / (end - start)

        cv2.putText(
            annotated,
            f"D455 + YOLOv8n | GPU | FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("D455 YOLO Depth Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()