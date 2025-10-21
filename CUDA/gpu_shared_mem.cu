#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10000
#define BLOCK_DIM 32


void fill_matrix_random(float* matrix, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            matrix[index] = (float)rand() / RAND_MAX * 1.0f;
        }
    }
}

__global__ void matrix_multiply_kernel(float* A, float* B, float* result, int width) {
    __shared__ float Ads[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bds[BLOCK_DIM][BLOCK_DIM];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by*BLOCK_DIM + ty;
    int col = bx*BLOCK_DIM + tx;
    
    if (row < width && col < width){
        float res = 0.0f;

        for (int m = 0; m < width/BLOCK_DIM; ++m) {
            // Load in a parallel manner.
            Ads[ty][tx] = A[row*width + (m*BLOCK_DIM + tx)];
            Bds[ty][tx] = B[(m*BLOCK_DIM + ty)*width + col];
            __syncthreads();
            for (int k = 0; k < BLOCK_DIM; ++k) {
                res += Ads[ty][k] * Bds[k][tx];
            }
            __syncthreads();
        }
        result[row * width + col] = res;
    }
}


int main() {
    clock_t start_time, end_time;
    srand(time(NULL));
    
    // Allocate matrices as 1D arrays
    float* matrixA = (float*)malloc(N * N * sizeof(float));
    float* matrixB = (float*)malloc(N * N * sizeof(float));
    float* result = (float*)malloc(N * N * sizeof(float));
    
    if (matrixA == NULL || matrixB == NULL || result == NULL) {
        printf("Failed!\n"); return 1;
    }
    
    // Fill matrices with random values
    fill_matrix_random(matrixA, N);
    fill_matrix_random(matrixB, N);

    // Allocate memory on GPU.
    float *Ad, *Bd, *Rd;
    int size = N * N * sizeof(float);
    cudaMalloc((void**)&Ad, size);
    cudaMalloc((void**)&Bd, size);
    cudaMalloc((void**)&Rd, size);

    // Move objects to GPU.
    cudaMemcpy(Ad, matrixA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Bd, matrixB, size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    // Size of each block (maximum 1024 threads per block). Set as multiple of 32 (warp scheduler)
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    // Grid of blocks 63 x 63 as 63 * 16 = 1008 > 1000
    dim3 gridSize(
        (N + blockSize.x - 1) / blockSize.x,
        (N + blockSize.y - 1) / blockSize.y
    );


    // Multiply matrices
    start_time = clock();
    // __managed__

    matrix_multiply_kernel<<<gridSize, blockSize>>>(Ad, Bd, Rd, N);
    //cudaDeviceSynchronize();
    end_time = clock();

    // Copy result back to Host.
    cudaMemcpy(result, Rd, size, cudaMemcpyDeviceToHost);

    // Release memory
    cudaFree(Ad); cudaFree(Bd); cudaFree(Rd);
    
    // Measure time
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Time: %f seconds\n", time_taken);
    
    // Free allocated memory
    free(matrixA);
    free(matrixB);
    free(result);
    
    return 0;
}
