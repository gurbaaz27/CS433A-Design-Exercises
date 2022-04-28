#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
#define NUM_THREADS_PER_BLOCK 256

__global__ void init_kernel (float *a, int span, int n)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int i;
    for (i = id * span; i < (id + 1) * span; i++)
        a[i] = (float)(i)/(n*n);
}

__global__ void matrix_vector_product_kernel (float *A, float *x, float *y, int span, int n)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
	int i, j;

	__shared__ float xs[NUM_THREADS_PER_BLOCK];

	for(i = 0; i < span; i++)
		y[i + id*span] = 0;

	for(i = 0; i < n/NUM_THREADS_PER_BLOCK; i++) {
	    xs[threadIdx.x] = x[i * NUM_THREADS_PER_BLOCK + threadIdx.x];
	    __syncthreads();
            for(int sp = 0; sp < span; sp++) {
                for(j = 0; j < NUM_THREADS_PER_BLOCK; j++) {
                    y[sp +id *span] += A[(id*span + sp) * n + i*NUM_THREADS_PER_BLOCK + j] * xs[j];
                }
            }
	    __syncthreads();
	}

}	

int main (int argc, char *argv[])
{
	float *A, *x, *y;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;

	if(argc != 3) {
		printf("Need n and t\n");
		exit(1);
    }

    int n = atoi(argv[1]), t = atoi(argv[2]);

    assert((t > 0) && ((t % 256) == 0) && (t & (t - 1)) == 0);
    //Assert that n is large enough
    assert(n > 0 && n % 256 == 0 && (n & (n - 1)) == 0);

    cudaMallocManaged((void**)&A, sizeof(float)*n*n);
    cudaMallocManaged((void**)&x, sizeof(float)*n);
    cudaMallocManaged((void**)&y, sizeof(float)*n);

	int device = -1;
	cudaGetDevice(&device);
	cudaMemAdvise(y, sizeof(float)*n, cudaMemAdviseSetPreferredLocation, device);

    if(t > n * n) 
		t = n * n;
	init_kernel<<<t/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(A, n*n/t, n);
    
	int t1 = t;
    if(t1 > n) 
		t1 = n;
	init_kernel<<<t1/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(x, n/t1, n);

	cudaDeviceSynchronize();

	gettimeofday(&tv0, &tz0);

	// dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	// dim3 dimGrid(BLOCKS_X, BLOCKS_Y);

	matrix_vector_product_kernel<<<t1/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(A, x, y, n/t1, n);
	cudaDeviceSynchronize();
	// printf("Span = %d\n", n/t1);

	gettimeofday(&tv2, &tz2);

    float error = 0;

    for(int i = 0; i < n; i++) {
        float x1 = 0;
        for(int j = 0; j < n; j++) x1 += A[i * n + j] * x[j];
        error += fabs(x1 - y[i]);
    }

    error /= n;

	printf("Average Error: %f, time: %ld microseconds\n", error, (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));

	return 0;
}
