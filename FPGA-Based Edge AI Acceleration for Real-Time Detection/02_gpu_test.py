from ultralytics import YOLO
import numpy as np
import time

model = YOLO("yolov8n.pt")
model.to("cuda")

img = np.random.randint(
    0, 255,
    (640, 640, 3),
    dtype=np.uint8
)

# warmup
for _ in range(20):
    model.predict(img, device="cuda", verbose=False)

times = []

for _ in range(100):
    start = time.time()

    model.predict(
        img,
        device="cuda",
        verbose=False
    )

    end = time.time()

    times.append(end - start)

avg = sum(times) / len(times)

print(f"Average latency: {avg*1000:.2f} ms")
print(f"FPS: {1/avg:.2f}")