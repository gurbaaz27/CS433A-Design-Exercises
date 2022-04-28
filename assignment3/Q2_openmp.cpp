#include <iostream>
#include <cmath>
#include <vector>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>

using namespace std;

int main(int argc, char *argv[]) 
{
	struct timeval tv0, tv1;
	struct timezone tz0, tz1;

    if(argc != 3) {
        cout << "Need n and t\n";
        exit(1);
    }

    int n = atoi(argv[1]), t = atoi(argv[2]);
    assert((t > 0) && (t & (t - 1)) == 0);
    // Assert that n is large enough
    assert((n > 0) && (n % 1024 == 0) && (n & (n - 1)) == 0);

    vector<vector<float>> A(n, vector<float>(n, 0));
    vector<float> x(n, 0);
    vector<float> y(n, 0);
    
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = (float)(i + j * n)/(n*n);
        }
    }
    
    for(int i = 0; i < n; i++) {
        x[i] = (float)(i)/(n*n);
    }

    gettimeofday(&tv0, &tz0);
    
    // row version
    int i, j;
    #pragma omp parallel for num_threads (t) private(i, j)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            y[i] += A[i][j] * x[j];
        }
    }

// // column version
// #pragma omp parallel num_threads (t)
// {
//     int i, j;
//     float ans = 0;
//     for(i = 0; i < n; i++) {
//         #pragma omp for //reduction (+:res[i])
//         for(j = 0; j < n; j++) {
//             ans += (x[j] * A[j][i]);
//         }
//     }        
// #pragma omp critical
//     {
//         for(i = 0; i < n; i++) {
//             y[i] += ans;
//             ans = 0;
//         }
//     }
// }

    gettimeofday(&tv1, &tz1);

    float error = 0;

    for(int i = 0; i < n; i++) {
        float x1 = 0;
        for(int j = 0; j < n; j++) 
            x1 += A[i][j] * x[j];
        error += fabs(x1 - y[i]);
    }

    error /= n;

    cout << "Size of matrix: " << n << "*" << n << "\n";
    cout << "Number of threads: " << t << "\n";
	cout << "Average Error: " << error << ", time: " << (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec) << " microseconds\n";
    
    return 0;
}
