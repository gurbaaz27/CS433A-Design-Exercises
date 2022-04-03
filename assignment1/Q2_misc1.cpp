#include <iostream>
#include <fstream>
#include <vector>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>
#include <math.h>

using namespace std;

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
    int n = 10000;
    vector<vector<long double>> l(n, vector<long double>(n, 0));
    //vector<long double> x(n, 0), y(n, 0);
    long double x[n], y[n];
    for(int i = 0; i < n; i++) {x[i] = 0; y[i] = 0;}
    for(int i = 0; i < n; i++) {
        for(int j = 0; j <= i; j++) l[i][j] = rand() % 10000 + 5;
    }
    for(int i = 0; i < n; i++) {
        x[i] = rand() % 10000;
	for(int j = 0; j <= i; j++) y[i] += l[i][j] * x[j];
    }
    vector<long double> ans(n);

    gettimeofday(&tv0, &tz0);
    cout << nThreads << endl;
#pragma omp parallel num_threads (nThreads)
    for(int i = 0; i < n; i++) {
#pragma omp for reduction (-:y[i])
        for(int j = 0; j < i; j++) {
	    y[i] -= l[i][j] * ans[j];
	    //cout << j << endl;
	}
	ans[i] = y[i] / l[i][i];
#pragma omp barrier
    }

    for(int i = 0; i < n; i++) assert(abs(ans[i] - x[i]) < 1e-4);
    //fin >> n;

    gettimeofday(&tv1, &tz1);
    for(int i = 0; i < n; i++) fout << ans[i] << " " << x[i] << endl;
    cout << "Time: " << (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec) << " microseconds\n";

    fin.close();
    fout.close();
    return 0;
}
