#include "sync_library.c"

void* solver(void* param) {
    int curr_sense = 0;
    int id = *(int*)param;
    for(int i = 0; i < N_barrier; i++) {
        Rev_Sense_Barrier(&curr_sense);
        // Tree_Barrier(id);
        // Central_POSIX_Barrier();
        // Tree_CV_Barrier(id);
        // pthread_barrier_wait(&barrier);
    }
}


int main(int argc, char *argv[]) {
	pthread_t *tid;
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

	if(argc != 2) {
		printf ("Need number of threads.\n");
		exit(1);
	}

	nThreads = atoi(argv[1]);
    MAX = LOG2(nThreads);

    for(int i = 0; i < nThreads; i++) {
        for(int j = 0; j < MAX; j++) flag[i][j] = 0;
    }

    Central_Posix_barr.counter = 0;
    pthread_mutex_init(&Central_Posix_barr.lock, NULL);
    pthread_cond_init(&Central_Posix_barr.cv, NULL);


    for(int i = 0; i < nThreads; i++) {
        for(int j = 0; j < MAX; j++) {
            Tree_CV_barr[i][j].flag = 0;
            pthread_mutex_init(&Tree_CV_barr[i][j].lock, NULL);
            pthread_cond_init(&Tree_CV_barr[i][j].cv, NULL);
        }
    }

    pthread_barrier_init(&barrier, NULL, nThreads);

	tid = (pthread_t*)malloc(nThreads * sizeof(pthread_t));
    int id[nThreads];
 	for(int i = 0; i < nThreads; i++) id[i] = i;

    pthread_attr_t attr;
    pthread_attr_init(&attr);

	gettimeofday(&tv0, &tz0);

	for(int i = 1; i < nThreads; i++) {
		pthread_create(&tid[i], &attr, solver, &id[i]);
   	}

    int curr_sense = 0;

    for(int i = 0; i < N_barrier; i++) {
        Rev_Sense_Barrier(&curr_sense);
        // Tree_Barrier(0);
        // Central_POSIX_Barrier();
        // Tree_CV_Barrier(0);
    //    pthread_barrier_wait(&barrier);
    }

    for(int i = 1; i < nThreads; i++) {
		pthread_join(tid[i], NULL);
	}

    gettimeofday(&tv1, &tz1);

    printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
}