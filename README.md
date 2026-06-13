# Hardware Accelerated Computer Vision using YOLOv8 and KV260

## Overview
This project evaluates CPU, GPU, and FPGA-based acceleration for real-time computer vision applications.

## Hardware
- NVIDIA RTX 5070 Ti GPU
- Intel CPU
- Intel RealSense D455
- AMD Xilinx KV260 Vision AI Starter Kit

## Features
- YOLOv8 object detection on CPU
- YOLOv8 object detection on GPU
- CPU vs GPU benchmarking
- RealSense D455 depth sensing
- Object distance estimation
- KV260 SmartCam deployment
- RTSP streaming
- FPGA-based edge AI acceleration

## Results
| Platform | Latency | FPS |
|-----------|----------|------|
| CPU | 25.43 ms | 39.40 |
| RTX 5070 Ti | 7.56 ms | 134.47 |
| KV260 FPGA | Real-time deployment | SmartCam Face Detection |

## Repository Structure
- scripts/
- results/
- report/
- recordings/
- kv260/

## Author
Cibiraj Esakkiraja
University of Turku