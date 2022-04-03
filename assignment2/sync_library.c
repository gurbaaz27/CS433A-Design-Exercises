#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <assert.h>
#include <semaphore.h>
#define MAX_SIZE 10000
#define N_lock 10000000
#define N_barrier 1000000

int nThreads;

/*******************
******* Locks ******
********************/

int x = 0, y = 0;

/* Shared Variables for Lamport */
unsigned char choosing[MAX_SIZE];
int ticket[MAX_SIZE];

/* Utility functions for Lamport */
int max(int a, int b) {
    return a > b ? a : b;
}

int comp_pair(int a1, int a2, int b1, int b2) {
    if(a1 < b1) return 1;
    if(a1 > b1) return 0;
    if(a2 < b2) return 1;
    return 0;
}

/* Acquire for Lamport */
void Acquire_Lamport(int tid) {
    choosing[64 * tid] = 1;
    asm("mfence":::"memory");
    int m = 0;
    for(int i = 0; i < nThreads; i++) m = max(m, ticket[16 * i]);
    ticket[16 * tid] = m + 1;
    asm("mfence":::"memory");
    choosing[64 * tid] = 0;
    asm("mfence":::"memory");

    for(int i = 0; i < nThreads; i++) {
        while(choosing[64 * i]);
        while(ticket[16 * i] && comp_pair(ticket[16 * i], i, ticket[16 * tid], tid));
    }
    return;
}

/* Release for Lamport */
void Release_Lamport(int tid) {
    asm("":::"memory");
    ticket[16 * tid] = 0;
    return;
}

/* Shared Variables for Spinlock */
int lock = 0;

/* Utility functions for Spinlock */
unsigned char CompareAndSet(int oldVal, int newVal, int *ptr) {
    int oldValOut;
    unsigned char result;
    asm("lock cmpxchgl %4, %1 \n setzb %0"
                :"=qm"(result),  "+m"(*ptr), "=a"(oldValOut)
                :"a"(oldVal),  "r"(newVal)
                : );

    return result;
}

/* Acquire for Spinlock */
void Acquire_SpinLock() {
    while(!CompareAndSet(0, 1, &lock));
    return;
}

/* Release for Spinlock */
void Release_SpinLock() {
    asm("":::"memory");
    lock = 0;
    return;
}


/* Shared Variables for Test-and-test-and-set */
int tts_lock = 0;

/* Utility functions for Test-and-test-and-set */
unsigned char TestAndSet(int *ptr) {
    int oldVal = 0;
    int oldValOut;
    int newVal = 1;
    unsigned char result;
    asm("lock cmpxchgl %4, %1 \n setzb %0"
                :"=qm"(result),  "+m"(*ptr), "=a"(oldValOut)
                :"a"(oldVal),  "r"(newVal)
                : );

    return 1 - result;
}

/* Acquire for Test-and-test-and-set */
void Acquire_TTS() {
    while(TestAndSet(&tts_lock)){
        while(tts_lock);
    }
    
    return;
}

/* Release for Test-and-test-and-set */
void Release_TTS() {
    asm("":::"memory");
    tts_lock = 0;
    return;
}

/* Shared Variables for Ticket Lock */
int _ticket = 0, release_count = 0;

/* Utility functions for Ticket Lock and Array Lock */
int FetchAndInc(int *ptr) {
    unsigned char result = 0;
    int oldVal, oldValOut, newVal;
    while(result != 1){
        oldVal = *ptr;
        oldValOut;
        newVal = *ptr+1;
        asm("lock cmpxchgl %4, %1 \n setzb %0"
                    :"=qm"(result),  "+m"(*ptr), "=a"(oldValOut)
                    :"a"(oldVal),  "r"(newVal)
                    : );
    }

    return oldVal;
}

/* Acquire for Ticket Lock */
void Acquire_TicketLock() {
    int curr_ticket = FetchAndInc(&_ticket);

    while(release_count != curr_ticket);

    return;
}

/* Release for Ticket Lock */
void Release_TicketLock() {
    asm("":::"memory");
    release_count++;

    return;
}


/* Shared Variables for Array Lock */
unsigned char available[MAX_SIZE];
int next_ticket = 0;

/* Acquire for Array Lock */
void Acquire_ArrayLock(int *ticket) {
    *ticket = FetchAndInc(&next_ticket);

    while(available[(*ticket % nThreads) * 64] != 1);

    return;
}

/* Release for Array Lock */
void Release_ArrayLock(int *ticket) {
    asm("":::"memory");
    available[(*ticket % nThreads) * 64] = 0;
    available[((*ticket + 1) % nThreads) * 64] = 1;

    return;
}

/* Shared variable for POSIX mutex */
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

/* Acquire for POSIX mutex */
void Acquire_POSIX_mutex() {
    pthread_mutex_lock(&mutex);
}

/* Release for POSIX mutex */
void Release_POSIX_mutex() {
    pthread_mutex_unlock(&mutex);
}

/* Shared variable for binary semaphore */
sem_t semaphore;

/* Acquire for binary semaphores */
void Acquire_Binary_Semaphore() {
    sem_wait(&semaphore);
}

/* Release for binary semaphores */
void Release_Binary_Semaphore() {
    sem_post(&semaphore);
}

/*******************
***** Barrier ******
********************/

/* Centralized sense-reversing barrier using busy-wait on flag */
struct Rev_Sense {
   int counter;
   pthread_mutex_t mutex;
   int flag;
} Rev_Sense_barr = {0, PTHREAD_MUTEX_INITIALIZER, 0};

void Rev_Sense_Barrier(int * local_sense) {
    *local_sense = !(*local_sense);
    pthread_mutex_lock(&Rev_Sense_barr.mutex);
    Rev_Sense_barr.counter++;
    if(Rev_Sense_barr.counter == nThreads) {
        pthread_mutex_unlock(&Rev_Sense_barr.mutex);
        Rev_Sense_barr.counter = 0;
        Rev_Sense_barr.flag = *local_sense;
    }
    else {
        pthread_mutex_unlock(&Rev_Sense_barr.mutex);

        for(int it =0 ;;it++){
            if(Rev_Sense_barr.flag == *local_sense) break;
        };
    }
    return;
}

/* Required variable for tree brriers */
int MAX; /* MAX = LOG2(nThreads); */

/* Tree barrier using busy-wait on flags */
int flag[16][4];

void Tree_Barrier(int pid) {
    unsigned int i, mask;

   for (i = 0, mask = 1; (mask & pid) != 0; ++i, mask <<= 1) {
      while (!flag[pid][i]);
      flag[pid][i] = 0;
   }
   if (pid < (nThreads - 1)) {
      flag[pid + mask][i] = 1; 
      while (!flag[pid][MAX- 1]);
      flag[pid][MAX - 1] = 0;
   }
   for (mask >>= 1; mask > 0; mask >>= 1) {
      flag[pid - mask][MAX-1] = 1; 
   }
}

/* Centralized barrier using POSIX condition variable */
struct Central_Posix {
    int counter; 
    pthread_mutex_t lock; 
    pthread_cond_t cv;
} Central_Posix_barr;

void Central_POSIX_Barrier() {
    pthread_mutex_lock(&Central_Posix_barr.lock);
    Central_Posix_barr.counter++;
    if (Central_Posix_barr.counter == nThreads) {
        Central_Posix_barr.counter = 0;
        pthread_cond_broadcast(&Central_Posix_barr.cv);
    }
    else pthread_cond_wait(&Central_Posix_barr.cv, &Central_Posix_barr.lock);
    pthread_mutex_unlock(&Central_Posix_barr.lock);
    return;
}

/* Tree barrier using POSIX condition variable */
struct Tree_CV {
    int flag;
    pthread_mutex_t lock; 
    pthread_cond_t cv;

} Tree_CV_barr[16][4];


int LOG2(int n) {
    int ans = 0;
    while(n != 1) {
        ans++;
        n >>= 1;
    }
    return ans;
}

void Tree_CV_Barrier(int pid) {
    unsigned int i, mask;

   for (i = 0, mask = 1; (mask & pid) != 0; ++i, mask <<= 1) {
      pthread_mutex_lock(&Tree_CV_barr[pid][i].lock);
      while (!Tree_CV_barr[pid][i].flag)pthread_cond_wait(&Tree_CV_barr[pid][i].cv, &Tree_CV_barr[pid][i].lock);
      Tree_CV_barr[pid][i].flag = 0;
      pthread_mutex_unlock(&Tree_CV_barr[pid][i].lock);
   }
   if (pid < (nThreads - 1)) {
      pthread_mutex_lock(&Tree_CV_barr[pid + mask][i].lock);
      Tree_CV_barr[pid + mask][i].flag = 1; 
      pthread_cond_broadcast(&Tree_CV_barr[pid + mask][i].cv);
      pthread_mutex_unlock(&Tree_CV_barr[pid + mask][i].lock);

      pthread_mutex_lock(&Tree_CV_barr[pid][MAX - 1].lock);
      while (!Tree_CV_barr[pid][MAX - 1].flag) pthread_cond_wait(&Tree_CV_barr[pid][MAX - 1].cv, &Tree_CV_barr[pid][MAX - 1].lock);
      Tree_CV_barr[pid][MAX - 1].flag = 0;
      pthread_mutex_unlock(&Tree_CV_barr[pid][MAX - 1].lock);
   }
   for (mask >>= 1; mask > 0; mask >>= 1) {
      pthread_mutex_lock(&Tree_CV_barr[pid - mask][MAX - 1].lock);
      Tree_CV_barr[pid - mask][MAX - 1].flag = 1; 
      pthread_cond_broadcast(&Tree_CV_barr[pid - mask][MAX - 1].cv);
      pthread_mutex_unlock(&Tree_CV_barr[pid - mask][MAX - 1].lock);
   }
}

/* POSIX barrier interface */
pthread_barrier_t barrier;

void POSIX_Barrier() {
    pthread_barrier_wait(&barrier);
}
