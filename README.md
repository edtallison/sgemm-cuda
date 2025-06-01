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
    - Threads within the same block e.g. `threadIds` (0, 0) and (0, 1) use the same column of B.
    - They each load the whole column from global memory. Hmmm this seems inefficient...

## 2. Global Memory Coalescing

- **Warps**
    - In execution, within a block, threads are grouped into "warps" of 32 threads.
    - Each streaming multiprocessor (SM) has four warp schedulers - physical cores that execute instructions.
    - Each warp is assigned to a warp scheduler, based on a consecutive `threadId` (x, y, z).
    - Threads with neighbouring `threadId` become part of the same warp.

- **Global Memory Coalescing**
    - Sequential memory acceses by threads in the same warp can be grouped and executed as one.
    - Important to keep in mind when optimising GMEM memory access.
    - For coalescing, the memory addresses need to be consecutive, but the within-warp accesses don't need to be consecutive.
    - GPU supports 32B, 64B, and 128B memory accesses.

- **Memory Access Pattern**
    (this part took me some time to get my head around)
    - In naive kernel, iterating threads with `threadIdx.x` (which aligns with consecutive `threadId`) actually leads to consecutive threads operating on consecutive rows of A, and the same row of B
    - If, instead, the threads operated on the same row of A but consecutive columns of B, this accessing of the B values could be coalesced.
    - This is achieved simply by changing the x and y position indices of the C element computed by each thread.
