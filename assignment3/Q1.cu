#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <sys/time.h>
// #include <curand.h>
// #include <curand_kernel.h>

#define TILE_SIZE 16
#define TOL 1e-5
#define ITER_LIMIT 1000

__device__ int count = 0, iters = 0;
__device__ float diff = 0;
__device__ volatile int barrier_flag = 0;
__device__ volatile int done = 0;

__global__ void init_kernel (float *a, int span, int n)
{
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int i;
    for (i = id * span; i < min((id + 1) * span, n*n); i++)
        a[i] =  ((float)(i % 100)/100.0); //curand_uniform(); // ((float)(curand_uniform() % 100)/100.0);
}

__global__ void gauss_siedel_kernal (float* a, int span, int nTiles_per_row, int n) 
{
	__shared__ float local_dif[TILE_SIZE*TILE_SIZE], local_dif2[TILE_SIZE*TILE_SIZE/32];
	int local_sense = 0, last_count;
	int index = threadIdx.x*TILE_SIZE + threadIdx.y;
	while(!done) {
		local_dif[index] = 0;
		for(int i = blockIdx.x*span; i < (blockIdx.x + 1)*span; i++) {
			int row = i/nTiles_per_row*TILE_SIZE + 1 + threadIdx.x, col = (i%nTiles_per_row)*TILE_SIZE + 1 + threadIdx.y;
			float temp = a[row*n + col];
			a[row*n + col] = 0.2 * (a[row*n + col] + a[row*n + col - 1] + a[row*n + col + 1] + a[(row-1)*n + col] + a[(row+1)*n + col]);
			local_dif[index] += fabs(temp - a[row*n + col]);
		}

		// atomicAdd(&diff, local_dif[index]);

        for (int i=warpSize/2; i>0; i=i/2) {
			local_dif[index] += local_dif[((index + i) % warpSize) + ((index)/warpSize)*warpSize];
			__syncwarp(0xffffffff); // No need to synchronize all threads, but this is low-cost
		}

        if ((index % warpSize) == 0) local_dif2[index/warpSize] = local_dif[index];

		local_sense = (local_sense ? 0 : 1);
		__syncthreads();

        if ((index/(TILE_SIZE*TILE_SIZE/32)) == 0) {
			for (int i=TILE_SIZE*TILE_SIZE/64; i>0; i=i/2) {
				local_dif2[index] += local_dif2[(index + i) % (TILE_SIZE*TILE_SIZE/32)];
				__syncwarp(0xffffffff); // No need to synchronize all threads, but this is low-cost
			}
			if (index == 0) atomicAdd(&diff, local_dif2[index]);
		}


		if (index == 0) {
			last_count = atomicAdd(&count, 1);
			if (last_count == (((n*n)/(span*TILE_SIZE*TILE_SIZE)) - 1)) {
				count = 0;
				if ((diff/(n*n) < TOL) || iters == ITER_LIMIT) {
					done = 1;
					printf("[%d] diff = %.10f\n", iters, diff/(n*n));
				}
				else {
					iters++;
					diff = 0;
				}
				barrier_flag = local_sense;
			}
		}
		while (barrier_flag != local_sense);
	}

}

int main (int argc, char *argv[])
{
	float *A;
	struct timeval tv0, tv2;
	struct timezone tz0, tz2;
        if(argc != 3) {
        	printf("Need n and t\n");
	       	exit(0);
    }
    int n = atoi(argv[1]), t = atoi(argv[2]);
    assert((t > 0) && ((t % 256) == 0) && (t & (t - 1)) == 0);
    //Assert that n is large enough
    assert(n > 0 && n % 256 == 0 && (n & (n - 1)) == 0);

	cudaMallocManaged((void**)&A, sizeof(float)*(n+2)*(n+2));

	int device = -1;
    cudaGetDevice(&device);
	// cudaMemAdvise(y, sizeof(float)*n, cudaMemAdviseSetPreferredLocation, device);

    if(t > n * n) t = n * n;
	init_kernel<<<t/256, 256>>>(A, ((n+2)*(n+2)+t-1)/t, n+2);
	cudaDeviceSynchronize();

	int numBlocksPerSm = 0, numBlocks;
	cudaDeviceProp deviceProp;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&deviceProp, device);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, gauss_siedel_kernal, TILE_SIZE*TILE_SIZE, 0); 
	numBlocks = deviceProp.multiProcessorCount*numBlocksPerSm;
	while ((numBlocks & (numBlocks - 1)) != 0) numBlocks--;
	if (t > (TILE_SIZE*TILE_SIZE*numBlocks)) t=TILE_SIZE*TILE_SIZE*numBlocks;

	dim3 dimBlock(TILE_SIZE, TILE_SIZE);


	gettimeofday(&tv0, &tz0);

	gauss_siedel_kernal<<<t/(TILE_SIZE*TILE_SIZE), dimBlock>>>(A, n*n/t, n/TILE_SIZE, n);
	cudaDeviceSynchronize();

	// dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	// dim3 dimGrid(BLOCKS_X, BLOCKS_Y);

	// printf("Span = %d\n", n/t1);

	gettimeofday(&tv2, &tz2);

	printf("Time: %ld microseconds\n", (tv2.tv_sec-tv0.tv_sec)*1000000+(tv2.tv_usec-tv0.tv_usec));
	return 0;
}
