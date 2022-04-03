#include "sync_library.c"

void* solver(void* param) {
    int id = *(int*)param;
    int array_lock_ticket;
    for(int i = 0; i < N_lock; i += 1) {
        Acquire_Lamport(id);
        // Acquire_SpinLock();
        // Acquire_TTS();
        // Acquire_TicketLock();
        // Acquire_ArrayLock(&array_lock_ticket);
        // Acquire_POSIX_mutex();
        // Acquire_Binary_Semaphore();

        assert (x == y);
        x = y + 1;
        y++;

        Release_Lamport(id);
        // Release_SpinLock();
        // Release_TTS();
        // Release_TicketLock();
        // Release_ArrayLock(&array_lock_ticket);
        // Release_POSIX_mutex();
        // Release_Binary_Semaphore();
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

	tid = (pthread_t*)malloc(nThreads * sizeof(pthread_t));
    int id[nThreads];
 	for(int i = 0; i < nThreads; i++) id[i] = i;

    for(int i = 1; i < MAX_SIZE; i++)
        available[i] = 0;
    available[0] = 1;

    sem_init(&semaphore, 0, 1);

    pthread_attr_t attr;
    pthread_attr_init(&attr);

	gettimeofday(&tv0, &tz0);

	for(int i = 1; i < nThreads; i++) {
		pthread_create(&tid[i], &attr, solver, &id[i]);
   	}

    int array_lock_ticket;

    for(int i = 0; i < N_lock; i += 1) {
        Acquire_Lamport(0);
        // Acquire_SpinLock();
        // Acquire_TTS();
        // Acquire_TicketLock();
        // Acquire_ArrayLock(&array_lock_ticket);
        // Acquire_POSIX_mutex();
        // Acquire_Binary_Semaphore();

        assert (x == y);
        x = y + 1;
        y++;

        Release_Lamport(0);
        // Release_SpinLock();
        // Release_TTS();
        // Release_TicketLock();
        // Release_ArrayLock(&array_lock_ticket);
        // Release_POSIX_mutex();
        // Release_Binary_Semaphore();
    }

    for(int i = 1; i < nThreads; i++) {
		pthread_join(tid[i], NULL);
	}

    gettimeofday(&tv1, &tz1);

    printf("Time: %ld microseconds\n", (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec));
}