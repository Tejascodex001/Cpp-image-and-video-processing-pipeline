cmake_minimum_required(VERSION 3.10)

project(CVTool)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(cv_tool main.cpp)

target_link_libraries(cv_tool ${OpenCV_LIBS})
target_include_directories(cv_tool PRIVATE ${OpenCV_INCLUDE_DIRS})
