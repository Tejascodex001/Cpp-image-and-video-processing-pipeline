# C++ Real-Time Image Processing & Telemetry Engine

**Status:** Active Development (Phase 2 Complete)

## Overview

This project is a systems engineering knowledge base for building real-time computer vision and AI pipelines from first principles. It bridges low-level C++ memory management, continuous hardware stream ingestion, and high-performance mathematical convolutions.

Unlike typical OpenCV tutorials that rely heavily on high-level abstractions and memory-heavy `.clone()` operations, this engine is built strictly on **zero-allocation architecture**. It is designed to process live video at maximum framerates by completely eliminating OS heap thrashing during the hot loop.

## Key Features

### Phase 1: Systems & Memory Foundations
* **Pure Mathematical Transforms:** Manual implementations of Grayscale conversion, 3x3 Box Blur, Zero-Padding, and Sobel Edge Detection.
* **Cache-Friendly Memory Access:** Bypasses `.at<>()` overhead by using raw row-pointer traversal (`.ptr`) for maximum cache-line utilization.
* **Safe Integer Accumulation:** Custom convolution kernels handle boundary conditions and prevent 8-bit unsigned integer overflow.

### Phase 2: Real-Time Streaming & Telemetry
* **Zero-Allocation Buffer Reuse:** Utilizes a strict Pass-by-Reference architecture with `cv::Mat::create()` and `.setTo()` to overwrite existing memory blocks, eliminating `malloc`/`new` calls at 30+ FPS.
* **Interactive State Machine:** Features runtime pipeline toggles. Subsystems can be enabled or disabled on the fly without halting the engine, utilizing Directed Acyclic Graph (DAG) dependency safety (e.g., safely computing Grayscale silently if Sobel requires it).
* **Dual-Clock Telemetry HUD:** Implements a Wall-Clock vs. Compute-Clock profiling system. Features a live Heads-Up Display (HUD) that renders exact micro-millisecond latencies for individual mathematical nodes, decoupling pure compute time from hardware auto-exposure limits.
* **Dynamic Diagnostic Dashboard:** Stitches multiple memory buffers (`hconcat`, `vconcat`) into a unified, real-time 4-pane UI.

## Build Instructions

**Prerequisites:**
* C++17 or higher
* CMake (>= 3.10)
* OpenCV (>= 4.x)

**Compilation:**
To achieve maximum performance, the project must be built in `Release` mode. This unleashes compiler optimizations like loop unrolling and vectorization.

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
## Usage

Execute the live streaming pipeline:

```bash
./live_video
```
**Interactive Controls:**
The pipeline acts as a live state machine. Use your keyboard to toggle rendering nodes in real-time while the video is playing:
* `g` - Toggle Grayscale compute/render
* `b` - Toggle 3x3 Box Blur compute/render
* `s` - Toggle Sobel Edge Detection compute/render
* `ESC` - Safely interrupt the hardware loop and exit

## Roadmap & Progression Architecture

This project follows a strict, non-negotiable progression path: Correctness → Memory Safety → Real-Time Streaming → Profiling → AI Inference.

- [x] **Phase 1:** Systems & Memory Foundations
- [x] **Phase 2:** Real-Time Streaming Systems & Telemetry
- [ ] **Phase 3:** AI Inference Integration (ONNX Runtime)
- [ ] **Phase 4:** Parallelism & Concurrency (Producer/Consumer queues)
- [ ] **Phase 5:** GPU Acceleration (CUDA)
- [ ] **Phase 6:** Autonomous Systems

## Architectural Philosophy (For Beginners)

If you are reading this code to learn systems programming, pay close attention to the `while(true)` loop inside `live_video.cpp`. You will notice that *no matrices are declared inside the loop*. 

In high-frequency systems (like streaming 30 frames per second), asking the Operating System for memory is extremely slow. This pipeline pre-allocates all required memory blocks when the program starts. As new frames arrive from the camera, the custom mathematical filters simply overwrite the existing bytes. This is how production-grade C++ engines maintain microsecond latencies.
