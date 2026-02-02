CUDA supports **up to 3D** indexing for both **grids** and **blocks**. This allows you to naturally map threads to 1D, 2D, or 3D data (e.g., vectors, images, volumes).

When launching a kernel, you specify:

```c
// dim3 is a CUDA built-in type with fields: x, y, z
dim3 threadsPerBlock(32, 32, 1);   // block has 32 * 32 * 1 threads
// If you omit z, it defaults to 1:
dim3 threadsPerBlock2(32, 32);     // same as (32, 32, 1)

// Grid dimensions (number of blocks)
dim3 numBlocks(2, 2, 1);           // grid has 2 * 2 * 1 blocks

kernel<<<numBlocks, threadsPerBlock>>>(...);
```

Inside the kernel, each thread computes its **global index**:

```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

---
## Bounds Checking (Critical)

> [!important] Rule of thumb  
> Every global memory access must be guarded by a bounds check against the array dimensions.

Example:

```c
if (row < N && col < K) {
    C[row * K + col] = ...;
}
```

Why this is necessary:
- You usually launch **more threads than elements**
- Extra threads must **not** access memory out of bounds
- Out-of-bounds access leads to **undefined behavior**, wrong results, or crashes

---

## Thread Block Size Limits

> [!warning] 1024 threads per block  
> **1024 is the maximum total number of threads per block**, not per dimension.

Valid examples (assuming the GPU supports the per-dimension sizes):
- `32 x 32 x 1 = 1024` (valid)
- `16 x 16 x 4 = 1024` (valid)
- `64 x 32 x 1 = 2048` (invalid: too many threads)

> [!important] Important points from the CUDA Programming Guide:
> - A block can be **1D, 2D, or 3D**
> - The product `blockDim.x * blockDim.y * blockDim.z` must be ≤ **1024**
> - Each **dimension** also has its own maximum size (architecture-dependent)
> - Grid dimensions can also be up to **3D** and have much larger limits (also architecture-dependent)

---
## Practical Guidelines

> [!example] Guidelines
> - Choose block sizes that are **multiples of 32 threads** (warp size)* when possible
> - Always add **bounds checks** for global memory accesses
> - Remember: **1024 is a limit on total threads per block**, not per dimension
> - Exact maximum dimensions depend on the GPU architecture