#include <stdio.h>
#include <stdlib.h>

void printMatricesSideBySide(int *A, int *B, int *C, int N, int M, int K)
{
    int maxRows = max(N, M);

    printf("      A (%dx%d)%*s      B (%dx%d)%*s      C (%dx%d)\n",
           N, M, M * 5 - 8, "",
           M, K, K * 5 - 8, "",
           N, K);

    for (int i = 0; i < maxRows; i++)
    {
        if (i < N)
        {
            for (int j = 0; j < M; j++)
                printf("%4d ", A[i * M + j]);
        }
        else
        {
            for (int j = 0; j < M; j++)
                printf("     ");
        }

        printf(" | ");

        if (i < M)
        {
            for (int j = 0; j < K; j++)
                printf("%4d ", B[i * K + j]);
        }
        else
        {
            for (int j = 0; j < K; j++)
                printf("     ");
        }

        printf(" | ");

        if (i < N)
        {
            for (int j = 0; j < K; j++)
                printf("%4d ", C[i * K + j]);
        }

        printf("\n");
    }

    printf("\n");
}

__global__ void multiplyMatrices_Kernel(int *A_d, int *B_d, int *C_d, int N, int M, int K)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < N && col < K)
    {
        int out = 0;

        for (int i = 0; i < M; i++)
        {
            out += A_d[row * M + i] * B_d[i * K + col];
        }

        C_d[row * K + col] = out;
    }
}

void multiplyMatrices_gpu(int *A, int *B, int *C, int N, int M, int K)
{
    int *A_d, *B_d, *C_d;
    cudaMalloc((void **)&A_d, sizeof(int) * N * M);
    cudaMalloc((void **)&B_d, sizeof(int) * M * K);
    cudaMalloc((void **)&C_d, sizeof(int) * N * K);

    cudaMemcpy(A_d, A, sizeof(int) * N * M, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, sizeof(int) * M * K, cudaMemcpyHostToDevice);

    dim3 numThreadsPerBlock(16, 16);
    dim3 numBlocks(ceil(K / (float)numThreadsPerBlock.x), ceil(N / (float)numThreadsPerBlock.y));
    multiplyMatrices_Kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, N, M, K);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(C, C_d, sizeof(int) * N * K, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    int N = 4, M = 4, K = 4;
    int *A = (int *)malloc(sizeof(int) * N * M);
    int *B = (int *)malloc(sizeof(int) * M * K);
    int *C = (int *)malloc(sizeof(int) * N * K);

    for (int i = 0; i < N * M; i++)
    {
        A[i] = rand() % 10;
    }

    for (int i = 0; i < M * K; i++)
    {
        B[i] = rand() % 10;
    }

    multiplyMatrices_gpu(A, B, C, N, M, K);
    printMatricesSideBySide(A, B, C, N, M, K);
}