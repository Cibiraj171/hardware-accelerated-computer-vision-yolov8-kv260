from ultralytics import YOLO
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

MODEL = "yolov8n.pt"
RUNS = 100

model = YOLO(MODEL)

img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

def benchmark(device_name, device):
    print(f"\nRunning {device_name} benchmark...")

    for _ in range(20):
        model.predict(img, device=device, verbose=False)

    times = []

    for i in range(RUNS):
        start = time.time()
        model.predict(img, device=device, verbose=False)
        end = time.time()

        latency = (end - start) * 1000
        fps = 1000 / latency

        times.append({
            "platform": device_name,
            "run": i,
            "latency_ms": latency,
            "fps": fps
        })

    return times

cpu_results = benchmark("CPU", "cpu")
gpu_results = benchmark("GPU_RTX_5070_Ti", "cuda")

all_results = cpu_results + gpu_results
df = pd.DataFrame(all_results)
df.to_csv("benchmark_cpu_gpu.csv", index=False)

summary = df.groupby("platform")[["latency_ms", "fps"]].mean()
summary.to_csv("summary_cpu_gpu.csv")

print("\nSummary:")
print(summary)

plt.figure()
summary["fps"].plot(kind="bar")
plt.ylabel("Average FPS")
plt.title("YOLOv8n CPU vs GPU FPS")
plt.tight_layout()
plt.savefig("fps_cpu_gpu.png", dpi=300)

plt.figure()
summary["latency_ms"].plot(kind="bar")
plt.ylabel("Average Latency (ms)")
plt.title("YOLOv8n CPU vs GPU Latency")
plt.tight_layout()
plt.savefig("latency_cpu_gpu.png", dpi=300)

print("\nSaved:")
print("benchmark_cpu_gpu.csv")
print("summary_cpu_gpu.csv")
print("fps_cpu_gpu.png")
print("latency_cpu_gpu.png")