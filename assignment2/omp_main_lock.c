#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
#define N 10000000

int nThreads;
int x = 0, y = 0;


int main(int argc, char *argv[]) {
	pthread_t *tid;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if(argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}
	
    nThreads = atoi(argv[1]);
    int i;

	gettimeofday(&tv0, &tz0);

	#pragma omp parallel num_threads (nThreads) private (i)
    {
        for(i=0; i<N; i++) {
        #pragma omp critical
            {
                assert (x == y);
                x = y + 1;
                y++;
            }
        }
    }

    gettimeofday(&tv1, &tz1);
    
    assert (x == y);
    assert (x == N * nThreads);

    printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
}