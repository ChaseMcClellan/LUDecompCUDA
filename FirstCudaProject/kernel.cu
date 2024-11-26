/*
* LU Decomposition CUDA example
* This is an example algorthim that implements 
* LU Decomp to solve a system of equations
* 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 4
#define IDX(i, j, n) ((i) * (n) + (j))

__global__ void lu_decomposition(float* A, float* L, float* U, int n) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    if (k >= n) return;

    for (int k = 0; k < n; ++k) {
        for (int j = k; j < n; ++j) {
            U[IDX(k, j, n)] = A[IDX(k, j, n)];
        }

        for (int i = k; i < n; ++i) {
            if (i == k)
                L[IDX(i, k, n)] = 1.0;
            else
                L[IDX(i, k, n)] = A[IDX(i, k, n)] / U[IDX(k, k, n)];
        }

        for (int i = k + 1; i < n; ++i) {
            for (int j = k + 1; j < n; ++j) {
                A[IDX(i, j, n)] -= L[IDX(i, k, n)] * U[IDX(k, j, n)];
            }
        }
    }
}

__global__ void forward_substitution(float* L, float* b, float* y, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n) return;

    y[i] = b[i];
    for (int j = 0; j < i; ++j) {
        y[i] -= L[IDX(i, j, n)] * y[j];
    }
}

__global__ void backward_substitution(float* U, float* y, float* x, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n) return;

    int idx = n - i - 1;
    x[idx] = y[idx];
    for (int j = idx + 1; j < n; ++j) {
        x[idx] -= U[IDX(idx, j, n)] * x[j];
    }
    x[idx] /= U[IDX(idx, idx, n)];
}

void print_matrix(float* mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", mat[IDX(i, j, n)]);
        }
        printf("\n");
    }
}

void print_vector(float* vec, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%f\n", vec[i]);
    }
}

int main() {
    int size = N * N * sizeof(float);
    int vector_size = N * sizeof(float);

    // Host matrices and vectors
    float A[N * N] = {
        4, 3, 2, 1,
        3, 2, 1, 1,
        2, 1, 1, 1,
        1, 1, 1, 1
    };
    float b[N] = { 10, 6, 4, 3 };
    float L[N * N] = { 0 };
    float U[N * N] = { 0 };
    float x[N] = { 0 };
    float y[N] = { 0 };

    float* d_A, * d_L, * d_U, * d_b, * d_x, * d_y;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_L, size);
    cudaMalloc((void**)&d_U, size);
    cudaMalloc((void**)&d_b, vector_size);
    cudaMalloc((void**)&d_x, vector_size);
    cudaMalloc((void**)&d_y, vector_size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, vector_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    lu_decomposition << <gridSize, blockSize >> > (d_A, d_L, d_U, N);

    forward_substitution << <gridSize, blockSize >> > (d_L, d_b, d_y, N);

    backward_substitution << <gridSize, blockSize >> > (d_U, d_y, d_x, N);

    cudaMemcpy(L, d_L, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, d_x, vector_size, cudaMemcpyDeviceToHost);

    printf("Matrix L:\n");
    print_matrix(L, N);
    printf("Matrix U:\n");
    print_matrix(U, N);
    printf("Solution Vector x:\n");
    print_vector(x, N);

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}


