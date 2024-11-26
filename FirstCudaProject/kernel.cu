/*
* LU Decomposition CUDA example
* This is an example algorthim that implements 
* LU Decomp to help solve a system of equations
* 
*/
#include <stdio.h>
#include <cuda.h>

#define N 4 //"nxn" size of matrix 
#define IDX(i, j, n) ((i) * (n) + (j))

//Kernel for LU decomposition with pivoting
__global__ void lu_decomposition(float* A, float* L, float* U, int n) {
    int k = threadIdx.x;

    for (int pivot = 0; pivot < n; ++pivot) {
        //Step 1:Partial Pivoting (done by a single thread)
        if (k == 0) {
            int maxRow = pivot;
            for (int i = pivot + 1; i < n; ++i) {
                if (fabs(A[IDX(i, pivot, n)]) > fabs(A[IDX(maxRow, pivot, n)])) {
                    maxRow = i;
                }
            }

            //swap rows
            if (maxRow != pivot) {
                for (int j = 0; j < n; ++j) {
                    float temp = A[IDX(pivot, j, n)];
                    A[IDX(pivot, j, n)] = A[IDX(maxRow, j, n)];
                    A[IDX(maxRow, j, n)] = temp;
                }
            }
        }
        __syncthreads(); //pivot is complete

        //Step 2:Compute L and U
        if (k > pivot && k < n) {
            L[IDX(k, pivot, n)] = A[IDX(k, pivot, n)] / A[IDX(pivot, pivot, n)];
            U[IDX(pivot, k, n)] = A[IDX(pivot, k, n)];
        }

        if (k == pivot) {
            U[IDX(pivot, pivot, n)] = A[IDX(pivot, pivot, n)];
            L[IDX(pivot, pivot, n)] = 1.0; //diagonal of L is 1
        }
        __syncthreads(); //check L and U are updated

        //Step 3: Update rows of A
        if(k > pivot && k < n) {
            for (int j = pivot + 1; j < n; ++j) {
                A[IDX(k, j, n)] -= L[IDX(k, pivot, n)] * U[IDX(pivot, j, n)];
            }
        }
        __syncthreads(); //check all rows are updated
    }
}

//helper to print a matrix
void print_matrix(float* mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", mat[IDX(i, j, n)]);
        }
        printf("\n");
    }
}

int main() {
    int size = N * N * sizeof(float);

    //matrix
    float A[N * N] = {
        4, 3, 2, 1,
        3, 2, 1, 1,
        2, 1, 1, 1,
        1, 1, 1, 1
    };
    float L[N * N] = { 0 };
    float U[N * N] = { 0 };

    //device matrix
    float* d_A, * d_L, * d_U;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_L, size);
    cudaMalloc((void**)&d_U, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    //launch kernel
    dim3 blockSize(N);
    lu_decomposition << <1, blockSize >> > (d_A, d_L, d_U, N);

    //copy results
    cudaMemcpy(L, d_L, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(U, d_U, size, cudaMemcpyDeviceToHost);

    //print
    printf("Matrix L:\n");
    print_matrix(L, N);
    printf("Matrix U:\n");
    print_matrix(U, N);

    cudaFree(d_A);
    cudaFree(d_L);
    cudaFree(d_U);

    return 0;
}
