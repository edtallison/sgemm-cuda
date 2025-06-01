Reimplementation of Simon Boehm's [CUDA SGEMM](https://github.com/siboehm/SGEMM_CUDA) kernels.

Following the [article](https://siboehm.com/articles/22/CUDA-MMM), for my learning :).

## Setup (TODO change to colab..)

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.


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
