#!/bin/bash
# setup_deps.sh - Downloads required external C++ libraries

echo "[SYSTEM] Downloading ONNX Runtime (Linux x64)..."
mkdir -p third_party && cd third_party

# Download and extract
wget -q --show-progress https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
tar -zxf onnxruntime-linux-x64-1.17.1.tgz
mv onnxruntime-linux-x64-1.17.1 onnxruntime
rm onnxruntime-linux-x64-1.17.1.tgz

echo "[SYSTEM] ONNX Runtime downloaded to third_party/onnxruntime/"
