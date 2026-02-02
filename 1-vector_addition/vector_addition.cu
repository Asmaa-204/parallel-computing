#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void vecAdd_kernel(int *x, int *y, int *z, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N)
    {
        z[i] = x[i] + y[i];
    }
}

void vecAdd_gpu(int *x, int *y, int *z, int N)
{
    // 1. allocate memory on gpu
    int *x_d, *y_d, *z_d;

    cudaMalloc((void **)&x_d, sizeof(int) * N);
    cudaMalloc((void **)&y_d, sizeof(int) * N);
    cudaMalloc((void **)&z_d, sizeof(int) * N);

    // 2. copy data to gpu
    cudaMemcpy(x_d, x, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, sizeof(int) * N, cudaMemcpyHostToDevice);

    // 3. call gpu kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    vecAdd_kernel<<<blocks, threadsPerBlock>>>(x_d, y_d, z_d, N);

    // 4. copy result back to cpu
    cudaMemcpy(z, z_d, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // 5. deallocate gpu memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main()
{
    int N = 10;
    int *x = (int *)malloc(sizeof(int) * N);
    int *y = (int *)malloc(sizeof(int) * N);
    int *z = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; i++)
    {
        x[i] = rand() % 100;
        y[i] = rand() % 100;
    }

    vecAdd_gpu(x, y, z, N);

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", x[i], y[i], z[i]);
    }

    free(x);
    free(y);
    free(z);
}