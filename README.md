Reimplementation of Simon Boehm's [CUDA SGEMM](https://github.com/siboehm/SGEMM_CUDA) kernels.

Following the [article](https://siboehm.com/articles/22/CUDA-MMM), for my learning :).

## Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edtallison/sgemm-cuda/blob/master/run_on_colab.ipynb)

Click the link above to open and run the project in a GPU-enabled Google Colab environment. No additional setup required.

# Notes
(also scattered throughout kernel code)

## 1. Naive

- **Three-level hierarchy of computation**
    - Grid, block, thread. Assume grid and block are 2D, thread is the atomic unit of computation.
    - Blocks can have up to 1024 threads.
    - Threads within the same block share memory (SMEM).

- **Grid and block indexing**
    - `gridDim` specifies dimensions of the grid i.e. rows and columns of blocks.
    - `blockDim` specifies dimensions of the block i.e. rows and columns of threads.
    - `blockIdx.x/y/z` specifies the block's position in the grid.
    - `threadIdx.x/y/z `specifies the thread's position in the block.
    - When used within a kernel, these vars are automatically assigned by the CUDA runtime.

- **Matrix multiplication**
    - Matrix multiplication: element ij of C is the dot product of row i of A and column j of B.
    - In this kernel, each thread computes one element of C. This can obviously be done in parallel so no synchronisation is required.

- **Kernel launch**
    - When the kernel is launched, we make the grid as big as necessary to cover all of C, depending on the block size.
    - The kernel execution is launched asynchronously i.e. the function call on the host (CPU) returns immediately.

- **Memory access pattern**
    - Threads within the same block e.g. ThreadIds (0, 0) and (0, 1) use the same column of B.
    - They each load the whole column from global memory. Hmmm this seems inefficient...
