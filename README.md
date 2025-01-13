# LU Decomposition with CUDA

## Description
This project demonstrates parallel programming using CUDA by implementing the LU decomposition algorithm.
The solution leverages GPU kernels to optimize performance and showcases the integration of linear algebra concepts with high-performance computing techniques.

## Features
- LU decomposition of square matrices.
- Forward and backward substitution for solving systems of linear equations.
- GPU acceleration for optimized computation.
- Numerical stability with partial pivoting.

## Requirements
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- Compiler supporting C/C++ (e.g., `gcc`, `nvcc`).

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/lu-decomp-cuda.git
2.  Navigate to the project directory:
    cd lu-decomp-cuda
3. Compile the code using nvcc:
   nvcc -o lu_decomp lu_decomp.cu
4. Run the compiled program
   ./lu_decomp

##Example Input
Matrix A:
4 3 2 1
3 2 1 1
2 1 1 1
1 1 1 1

Vector b:
10 6 4 3

##Example Output
Solution Vector x:
1.000000
2.000000
-1.000000
3.000000

Matrix L:
1.000000 0.000000 0.000000 0.000000
0.750000 1.000000 0.000000 0.000000
0.500000 0.333333 1.000000 0.000000
0.250000 0.166667 0.500000 1.000000

Matrix U:
4.000000 3.000000 2.000000 1.000000
0.000000 -0.250000 -0.500000 -0.250000
0.000000 0.000000 -0.166667 -0.166667
0.000000 0.000000 0.000000 -0.083333

##Notes
The matrix size N is currently set to 4x4. Modify the #define N in the code for different sizes.
Ensure the input matrix is non-singular for proper LU decomposition.

Developed by Chase McClellan (Chat-GPT 4o created readme)
