title: "Native Python CUDA for Image Convolution"
date: 2025-04-28
description: "A practical guide to writing GPU kernels in pure Python using NVIDIA's native CUDA Python support."
author: Girish Kumar
tags: ["CUDA", "Python", "GPU", "Numba", "Image Processing", "AI", "2025"]

Introduction

In 2025, NVIDIA officially announced native Python support for CUDA, opening the doors for developers to write GPU code directly in Python — without needing any C++.

The first announcements about moving toward "Python-first CUDA" were discussed conceptually by NVIDIA engineers at GTC 2024.However, the real official release — usable for developers (i.e., cuda-python, cuda.core 0.2.0) — happened in March–April 2025.

This blog explains how I used this new capability to build a simple but powerful GPU-accelerated image convolution engine — fully inside Python — and benchmark it against CPU performance.

You’ll learn how to set up the environment, write a custom 2D convolution kernel, and launch it on a GPU step-by-step.

Environment Setup

Step 1: Enable GPU on Google Colab

Go to Runtime > Change runtime type > Hardware accelerator > GPU.

Step 2: Install CUDA Python Support

!pip install cuda-python numba numpy matplotlib

import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

Writing a GPU Kernel in Pure Python

In Native Python CUDA, we use the @cuda.jit decorator from Numba to define GPU kernels.These kernels are JIT-compiled into PTX and executed on the GPU directly.

Key Notes:

Use cuda.grid(2) to get the thread’s global (x, y) position.

Always check that your thread indices are within image bounds.

Each thread typically computes one output pixel.

@cuda.jit
def increment_2d(arr):
    x, y = cuda.grid(2)
    if x < arr.shape[0] and y < arr.shape[1]:
        arr[x, y] += 1

Example: 2D Image Convolution on GPU

Step 1: Prepare Image and Filter

img = np.zeros((64, 64), dtype=np.float32)
img[20:44, 20:44] = 1.0  # white square

kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
], dtype=np.float32)

output = np.zeros_like(img)

Step 2: Write the Convolution Kernel

@cuda.jit
def conv2d_kernel(input_img, output_img, filt, filt_width):
    x, y = cuda.grid(2)
    h, w = input_img.shape
    r = filt_width // 2
    if x < h and y < w:
        if x < r or x >= h - r or y < r or y >= w - r:
            return
        accum = 0.0
        for i in range(filt_width):
            for j in range(filt_width):
                px = x + i - r
                py = y + j - r
                accum += input_img[px, py] * filt[i, j]
        output_img[x, y] = accum

Step 3: Copy Data to GPU and Launch Kernel

d_input = cuda.to_device(img)
d_output = cuda.to_device(output)
d_kernel = cuda.to_device(kernel)

threads_per_block = (16, 16)
blocks_x = int(np.ceil(img.shape[0] / threads_per_block[0]))
blocks_y = int(np.ceil(img.shape[1] / threads_per_block[1]))
blocks_per_grid = (blocks_x, blocks_y)

conv2d_kernel[blocks_per_grid, threads_per_block](d_input, d_output, d_kernel, kernel.shape[0])

result = d_output.copy_to_host()

Step 4: Visualize Results

plt.imshow(result, cmap='gray')
plt.title("GPU Convolution Output")
plt.show()

Native Python CUDA vs. CuPy or Traditional CUDA

Aspect

CuPy

Traditional CUDA

Native Python CUDA

Base Language

Python

C++

Python

Compilation

Precompiled libraries

Manual .cu and nvcc

JIT-compiled from Python

Kernel Writing

Use strings or APIs

Full C++ syntax

Full Python syntax

Flexibility

Moderate

Very High

High

Ease of Use

Easy

Hard

Easy

Best Practices for CUDA Python

Use 16x16 thread blocks for 2D image tasks.

Always check array bounds to avoid memory errors.

Keep data on GPU if running multiple steps.

Start with small images to debug kernels.

Use profiling tools for performance bottlenecks.

Conclusion

Native Python CUDA represents a huge leap in GPU programming accessibility.
Without touching C++, I wrote and launched a real GPU kernel to perform 2D image convolution — all inside Python.

This new stack is exciting for AI, ML, and HPC developers — unlocking custom GPU workflows without the complexity of C++.

Highly recommend exploring cuda-python and numba in 2025 if you're building compute-heavy workflows in Python.

About the Author

I'm Girish Kumar, passionate about AI, ML, and cutting-edge GPU programming.In this project, I explored NVIDIA’s new Native Python CUDA approach to build a custom image convolution engine fully in Python — no C++ required.

Connect with me on LinkedIn for more such explorations in AI + HPC!