#include <stdio.h>

__global__ void helloCUDA() {
    printf("Hello from the GPU thread %d\n", threadIdx.x);
}

int main() {
    // Launch kernel
    helloCUDA << <1, 10 >> > ();
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
