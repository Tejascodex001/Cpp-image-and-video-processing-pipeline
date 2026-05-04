# C++ Image Processing Pipeline (OpenCV)

## Overview

A low-level image processing pipeline implemented in C++ using OpenCV.
Includes manual implementations of grayscale conversion, blur (3×3), padding, and Sobel edge detection using direct memory access (`.ptr`).

## Features

* Custom grayscale (no cvtColor)
* Manual blur filter (3×3 kernel)
* Zero padding (grayscale + color)
* Sobel edge detection (manual convolution)
* Pointer-based memory access for performance

## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Run

```bash
./cv_tool <image_path>
```

## Notes

* Designed for learning systems-level image processing
* Functions structured as a processing pipeline (no side effects)

