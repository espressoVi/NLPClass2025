#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2000


void fill_matrix_random(float* matrix, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            matrix[index] = (float)rand() / RAND_MAX * 1.0f;
        }
    }
}

void print_matrix(float* matrix, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            printf("%8.2f ", matrix[index]);
        }
        printf("\n");
    }
}

// Multiplies two N×N matrices, matrices are stored in row-major order as 1D arrays
void matrix_multiply(float* A, float* B, float* result, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            int result_index = i * width + j;
            result[result_index] = 0.0f;
            
            float res = 0;
            for (int k = 0; k < width; k++) {
                int a_index = i * width + k;
                int b_index = k * width + j;
                res += A[a_index] * B[b_index];
            }
            result[result_index] = res;
        }
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
    
    // printf("Matrix A:\n");
    // print_matrix(matrixA, N);
    // printf("\nMatrix B:\n");
    // print_matrix(matrixB, N);
    
    // Multiply matrices
    start_time = clock();
    matrix_multiply(matrixA, matrixB, result, N);
    end_time = clock();
    
    // printf("\nResult (A × B):\n");
    // print_matrix(result, N);

    // Measure time
    double time_taken = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Time: %f seconds\n", time_taken);
    
    // Free allocated memory
    free(matrixA);
    free(matrixB);
    free(result);
    
    return 0;
}
