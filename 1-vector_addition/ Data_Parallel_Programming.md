## Types of Parallelism

### 1. Task Parallelism
- Different operations on the same or different data.
- An application may consist of multiple tasks (dependent or independent) that can run in parallel.
### 2. Data Parallelism
- The same operation is applied to different pieces of data.
- Example: computing pixel values for an image, vector addition, matrix operations.
- This model is **especially suitable for GPUs**.
---
## System Organization (CPU vs GPU)

- The CPU and GPU have **separate memory spaces**.
- The CPU **cannot directly access GPU memory**, and the GPU **cannot directly access CPU memory**.
- Data must be transferred explicitly between them using an interconnect such as:
    - **PCIe** (general, common)
    - **NVLink** (NVIDIA-specific, higher bandwidth)

So the typical workflow is:
1. Allocate memory on the GPU
2. Copy data from CPU → GPU
3. Run computation on the GPU
4. Copy results from GPU → CPU
5. Free GPU memory

---

## Sequence of Offloading Computation from CPU to GPU

### 1. Allocate Memory on the GPU

```c
cudaError_t cudaMalloc(void **devPtr, size_t size);
```

- `devPtr`: pointer to where the device pointer will be stored
- `size`: number of bytes to allocate
- Returns a `cudaError_t` indicating success or failure

Example:

```c
float *x_d;  // _d convention: variable lives on the device (GPU)
cudaMalloc((void**)&x_d, N * sizeof(float));
```

---
### 2. Copy Data Between CPU and GPU

```c
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
```

`kind` specifies the direction:
- `cudaMemcpyHostToDevice`
- `cudaMemcpyDeviceToHost`
- `cudaMemcpyHostToHost`
- `cudaMemcpyDeviceToDevice`

---
### 3. Perform Computation on the GPU (Launch a Kernel)

> [!info] kernel:
> a function that runs on the GPU and is executed by many threads in parallel.

Kernel launch syntax:

```c
vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(...);
```

- This creates a **grid** of thread blocks.
- Each block contains `numThreadsPerBlock` threads.
- Each thread executes the same kernel function.

Typical block sizes are:
- 128, 256, 512, or 1024 (MAX) threads (usually powers of two)
- The exact choice depends on the GPU and the problem.
- e.g.
 ``` c
int threadsPerBlock = 256;
int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
 ```

---
### 4. Copy Results Back from GPU to CPU

```c
cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);
```

---

### 5. Deallocate GPU Memory

```c
cudaError_t cudaFree(void *devPtr);
```

Example:

```c
cudaFree(x_d);
```

---
## Grid, Blocks, and Threads

- A **grid** is a collection of thread blocks.
- A **block** is a group of threads that:
    - Can cooperate using shared memory
    - Can synchronize with each other
- Threads in **different blocks cannot synchronize directly**.

All threads in a grid execute the **same kernel function**, following the **SPMD** model:

> [!info] **SPMD: Single Program, Multiple Data**  
> Multiple threads run the same program but operate on different data elements.

---
## Kernel Functions

A kernel is declared using the `__global__` qualifier:

```c
__global__ void vecadd_kernel(...) {
    // GPU code
}
```

- `__global__` means:
    - Callable from the **CPU**
    - Runs on the **GPU**

Each thread can compute its global index:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockDim` is a built-in variable containing the block dimensions (`x`, `y`, `z`)
- `blockIdx` and `threadIdx` identify the block and thread

---
## Handling Non-Divisible Sizes (Boundary Check)

If the total number of threads is **greater than N**, extra threads must be disabled:

```c
if (i < N) {
    // safe to access element i
}
```

This prevents out-of-bounds memory access.

---
## CUDA File and Compilation Model

- CUDA source files use the `.cu` extension.
- They contain both:
    - Host (CPU) C/C++ code
    - Device (GPU) code
    
### NVCC (NVIDIA CUDA Compiler)

`nvcc` splits the code into:

1. **Host code** → compiled by the host C/C++ compiler → runs on CPU    
2. **Device code (`__global__`, `__device__`)** → compiled to PTX (virtual ISA) → just-in-time `JIT` compiled → runs on GPU

Compile example:

```bash
nvcc vecadd.cu -o vecadd
```

---
## Function Qualifiers
| Qualifier     | Callable From | Runs On | Purpose |
|---------------|---------------|---------|---------|
| `__global__`  | CPU           | GPU     | Defines a kernel function launched from the host |
| `__device__`  | GPU           | GPU     | Defines a function callable only from device code |
| `__host__`    | CPU           | CPU     | Defines a function that runs on the host (default) |

You can combine them:

```c
__host__ __device__ float f(float a, float b) {
    return a + b;
}
```

This creates **two versions** of the function:
- One compiled for the CPU
- One compiled for the GPU

---
## Asynchronous Execution

- Kernel launches are **asynchronous by default** (CPU continues executing while the GPU runs the kernel)
- To wait for the GPU to finish:

```c
cudaDeviceSynchronize(); // returns cudaError_t
```

---
## Error Handling

Most CUDA runtime functions return `cudaError_t`:
```c
cudaError_t err = cudaMalloc(...);
if (err != cudaSuccess) {
    // handle error
}
```

You can also get a readable error string:
```c
printf("CUDA error: %s\n", cudaGetErrorString(err));
```

---