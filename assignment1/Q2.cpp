#include <iostream>
#include <fstream>
#include <vector>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <iomanip> 

using namespace std;

// void initialiseInput(vector <vector<long double>> &L, long double y[], vector <long double> &ac, int n){
//     for(int i = 0; i < n; i++) {
//         for(int j = 0; j <= i; j++) 
//             L[i][j] = rand() % 10000 + 5;
//     }
//     for(int i = 0; i < n; i++) {
//         ac[i] = rand() % 10000;
// 	    for(int j = 0; j <= i; j++) 
//             y[i] += L[i][j] * ac[j];
//     }
// }

int main(int argc, char *argv[]) {
    if(argc != 4) {
        cout << "Need 3 arguments.\n";
        exit(1);
    }
    int nThreads = atoi(argv[3]);
    ifstream fin;
    fin.open(argv[1]);
    ofstream fout;
    fout.open(argv[2]);
    assert(fin);
    assert(fout);

    struct timeval tv0, tv1;
    struct timezone tz0, tz1;
    // int n = 10000;
    int n;
    // fin << n << "\n";
    fin >> n;
    vector<vector<long double>> L(n, vector<long double>(n, 0));
    vector<long double> x(n, 0);
    // vector<long double> ac(n, 0);
    long double y[n] = {0};
    // initialiseInput(L, y, ac, n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i + 1; j++) {
            // fin << L[i][j] << " ";
            fin >> L[i][j];
	    }
        // fin << "\n";
    }
    for(int i = 0; i < n; i++){
        // fin << fixed << setprecision(6) << y[i] << " ";
        fin >> y[i];
    }

    gettimeofday(&tv0, &tz0);
    
    int j;

#pragma omp parallel num_threads (nThreads)
    for(int i = 0; i < n; i++) {
#pragma omp for reduction (-:y[i])
        for(j = 0; j < i; j++) {
            y[i] -= (L[i][j] * x[j]);
        }
        x[i] = y[i] / L[i][i];
#pragma omp barrier
    }

// #pragma omp parallel num_threads (nThreads)
//     for(int i=0; i<n; i++){
//         x[i] = y[i]/L[i][i];
// #pragma omp for 
//         for(int j=i+1; j<n; j++){
//             y[j] -= L[j][i]*x[i];
//         }
//     }

    gettimeofday(&tv1, &tz1);

    for(int i = 0; i < n; i++) 
        fout << x[i] << " ";
    fout << endl;
    cout << "Size of matrix: " << n << "*" << n << "\n";
    cout << "Number of threads: " << nThreads << "\n";
    cout << "Time: " << (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec) << " microseconds\n";
    fin.close();
    fout.close();
    return 0;
}
