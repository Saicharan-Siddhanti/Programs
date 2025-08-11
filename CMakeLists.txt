cmake_minimum_required(VERSION 3.10)
project(custom_profiler LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable CUDA (optional)
set(USE_CUDA OFF CACHE BOOL "Disable CUDA")
set(CMAKE_DISABLE_FIND_PACKAGE_CUDA TRUE)

# Make sure CMAKE_PREFIX_PATH is set
if(NOT CMAKE_PREFIX_PATH)
    message(FATAL_ERROR "CMAKE_PREFIX_PATH is not set. Please run: cmake .. -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')")
endif()

# Find PyTorch
find_package(Torch REQUIRED PATHS ${CMAKE_PREFIX_PATH})

# Find pybind11
find_package(pybind11 REQUIRED)

# Create shared library for Python extension
pybind11_add_module(custom_profiler kineto_no_autograd.cpp)

# Link against Torch
target_link_libraries(custom_profiler PRIVATE "${TORCH_LIBRARIES}")

# Include Torch headers
target_include_directories(custom_profiler PRIVATE ${TORCH_INCLUDE_DIRS})
