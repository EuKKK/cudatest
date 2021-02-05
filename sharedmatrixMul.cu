#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define M 1024
#define K 1024
#define N 1024

#define BLOCK_SIZE 32  //block size ,each thread to calucate each bloc

void initial(double* array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (double)(rand() % 10 + 1);
	}
}

void printMatrix(double* array, int row, int col)
{
	double* p = array;
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			printf("%10lf", p[x]);
		}
		p = p + col;
		printf("\n");
	}
	return;
}


void  multiplicateMatrixOnHost(double* array_A, double* array_B, double* array_C, int M_p, int K_p, int N_p)
{
	for (int i = 0; i < M_p; i++)
	{
		for (int j = 0; j < N_p; j++)
		{
			double sum = 0;
			for (int k = 0; k < K_p; k++)
			{
				sum += array_A[i * K_p + k] * array_B[k * N_p + j];
			}
			array_C[i * N_p + j] = sum;
		}
	}

}

__global__ void multiplicateMatrixOnDevice(double* array_A, double* array_B, double* array_C, int M_p, int K_p, int N_p)
{
	int ix = threadIdx.x + blockDim.x * blockIdx.x;//col  number
	int iy = threadIdx.y + blockDim.y * blockIdx.y;//row number

	if (ix < N_p && iy < M_p)
	{
		double sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += array_A[iy * K_p + k] * array_B[k * N_p + ix];
		}
		array_C[iy * N_p + ix] = sum;
	}
}

// Compute C = A * B
__global__ void matrixMultiplyShared(double* A, double* B, double* C,
	int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP

	__shared__ double sharedM[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double sharedN[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;


	float Csub = 0.0;

	for (int i = 0; i < (int)(ceil((double)numAColumns / BLOCK_SIZE)); i++)
	{

		if (i * BLOCK_SIZE + tx < numAColumns && row < numARows)
			sharedM[ty][tx] = A[row * numAColumns + i * BLOCK_SIZE + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i * BLOCK_SIZE + ty < numBRows && col < numBColumns)
			sharedN[ty][tx] = B[(i * BLOCK_SIZE + ty) * numBColumns + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();


		for (int j = 0; j < BLOCK_SIZE; j++)
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}


	if (row < numCRows && col < numCColumns)
		C[row * numCColumns + col] = Csub;

}


int main(int argc, char** argv)
{
	clock_t start = 0, finish = 0;
	double time;

	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;


	double* h_A, * h_B, * hostRef, * deviceRef;
	h_A = (double*)malloc(Axy * sizeof(double));
	h_B = (double*)malloc(Bxy * sizeof(double));

	int nBytes = M * N * sizeof(double);
	hostRef = (double*)malloc(Cxy * sizeof(double));
	deviceRef = (double*)malloc(Cxy * sizeof(double));

	initial(h_A, Axy);
	printf("\n");
	printf("Matrix_A: (%dx%d)\n", M, K);
	//printMatrix(h_A, M, K);
	initial(h_B, Bxy);
	printf("Matrix_B: (%dx%d)\n", K, N);
	//printMatrix(h_B, K, N);

	start = clock();
	multiplicateMatrixOnHost(h_A, h_B, hostRef, M, K, N);
	finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;

	printf("\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnHost \n");
	printf("------------------------------------------------------------------------------------\n");

	printf("Matrix_hostRef: (%dx%d)  CPUtime:%lfs\n", M, N, time);
	//printMatrix(hostRef, M, N);

	double* d_A, * d_B, * d_C;
	cudaMalloc((void**)&d_A, Axy * sizeof(double));
	cudaMalloc((void**)&d_B, Bxy * sizeof(double));
	cudaMalloc((void**)&d_C, Cxy * sizeof(double));

	cudaMemcpy(d_A, h_A, Axy * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bxy * sizeof(double), cudaMemcpyHostToDevice);


	printf("\n\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnDevice \n");
	printf("------------------------------------------------------------------------------------\n");




	int dimx = 32;
	int dimy = 32;
	dim3 block(dimx, dimy);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
	//	dim3 grid(1, 1);

	cudaEvent_t gpustart, gpustop;
	float elapsedTime = 0.0;
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);
	matrixMultiplyShared << < grid, block >> > (d_A, d_B, d_C, M, K, K, N, M, N);
	//	printf("   multiplicateMatrixOnDevice<<<(%d,%d),(%d,%d)>>>", grid.x, grid.y, block.x, block.y);
	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);


	cudaMemcpy(deviceRef, d_C, Cxy * sizeof(double), cudaMemcpyDeviceToHost);
	printf("Matrix_deviceRef: (%dx%d)  <<<(%d,%d),(%d,%d)>>>  GPUtime:%fms\n",
		M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);
	//printMatrix(deviceRef, M, N);

	bool ret = true;
	for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			if (hostRef[i*M+j] != deviceRef[i*M+j])
				ret = false;
		}
	}

	printf("%s\n", ret == true ? "pass" : "false");

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(hostRef);
	free(deviceRef);

	cudaDeviceReset();

	return (0);
}

