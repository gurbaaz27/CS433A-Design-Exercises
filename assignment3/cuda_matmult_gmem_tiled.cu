#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>

#define ROWS_A (1 << 13)
#define COLS_A (1 << 13)
#define ROWS_B COLS_A
#define COLS_B (1 << 13)
#define ROWS_C ROWS_A
#define COLS_C COLS_B

#define TILE_SIZE 16
#define THREADS_PER_BLOCK_X TILE_SIZE
#define THREADS_PER_BLOCK_Y TILE_SIZE
#define BLOCKS_X (COLS_C/THREADS_PER_BLOCK_X)
#define BLOCKS_Y (ROWS_C/THREADS_PER_BLOCK_Y)

__global__ void init_kernel (float *a)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;

	a[id] = (float)id/(ROWS_C*COLS_C);
}

__global__ void matmult_kernel (float *a, float *b, float *c)
{
	int c_row = blockIdx.y*blockDim.y + threadIdx.y;
	int c_col = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j;
	float x = 0;
	
	for (i=0; i<COLS_A/TILE_SIZE; i++) {
		for (j=0; j<TILE_SIZE; j++) {
			x += (a[c_row*COLS_A + i*TILE_SIZE + j]*b[c_col + COLS_B*(i*TILE_SIZE + j)]);
		}
		__syncthreads();
	}

	c[c_row*COLS_C + c_col] = x;
}	

int main (int argc, char *argv[])
{
	int i;
	float *a, *b, *c;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;

	cudaMallocManaged((void**)&a, sizeof(float)*ROWS_A*COLS_A);
	cudaMallocManaged((void**)&b, sizeof(float)*ROWS_B*COLS_B);
	cudaMallocManaged((void**)&c, sizeof(float)*ROWS_C*COLS_C);

	int device = -1;
        cudaGetDevice(&device);
	cudaMemAdvise(c, sizeof(float)*ROWS_C*COLS_C, cudaMemAdviseSetPreferredLocation, device);

	init_kernel<<<ROWS_A*COLS_A/1024, 1024>>>(a);
	init_kernel<<<ROWS_B*COLS_B/1024, 1024>>>(b);

	cudaDeviceSynchronize();

	gettimeofday(&tv0, &tz0);

	dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	dim3 dimGrid(BLOCKS_X, BLOCKS_Y);

	matmult_kernel<<<dimGrid, dimBlock>>>(a, b, c);
	cudaDeviceSynchronize();

	gettimeofday(&tv2, &tz2);

	int rowC = random() % ROWS_C;
	int colC = random() % COLS_C;

	float x = 0;

	for (i=0; i<COLS_A; i++) x += a[rowC*COLS_A + i]*b[colC + COLS_B*i];
	float error = fabs(c[rowC*COLS_C + colC] - x);
	
	printf("Error: %f, computed value: %f, actual value: %f, time: %ld microseconds\n", error, c[rowC*COLS_C + colC], x, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}
