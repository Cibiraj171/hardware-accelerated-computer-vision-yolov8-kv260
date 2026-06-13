import cv2, time, csv, os
import numpy as np

os.makedirs("results", exist_ok=True)

img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

times = []
for i in range(100):
    start = time.time()
    resized = cv2.resize(img, (320, 320))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    end = time.time()
    times.append((end - start) * 1000)

avg_latency = sum(times) / len(times)
fps = 1000 / avg_latency

print("KR260 CPU baseline")
print(f"Average latency: {avg_latency:.2f} ms")
print(f"FPS: {fps:.2f}")

with open("results/kr260_cpu_baseline.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["platform", "latency_ms", "fps"])
    writer.writerow(["KR260_ARM_CPU", avg_latency, fps])
