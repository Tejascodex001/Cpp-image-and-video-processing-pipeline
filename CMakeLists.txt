cmake_minimum_required(VERSION 3.10)
project(CV_Pipeline)

# --- Find OpenCV ---
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- ONNX Runtime Integration (Local) ---
set(ORT_DIR "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
include_directories(${ORT_DIR}/include)
link_directories(${ORT_DIR}/lib)

# Executable 1: Your static image tester
add_executable(cv_tool main.cpp vision_filters.cpp)
target_link_libraries(cv_tool ${OpenCV_LIBS})

# Executable 2: Your real-time video loop
add_executable(live_video live_video.cpp vision_filters.cpp)
# Link both OpenCV and the ONNX Runtime to the live engine
target_link_libraries(live_video ${OpenCV_LIBS} onnxruntime)
