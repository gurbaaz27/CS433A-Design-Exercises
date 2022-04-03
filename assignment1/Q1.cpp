#include <iostream>
#include <fstream>
#include <vector>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>
#include <assert.h>

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
    int n;
    fin >> n;
    vector<vector<int>> d(n, vector<int>(n));
    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            fin >> d[i][j];
            d[j][i] = d[i][j];
	    }
    }
    int m = 1 << n;
    vector<vector<int>> mask(n + 1), dp(m, vector<int>(n, INT_MAX)), trans(m, vector<int>(n));
   
    gettimeofday(&tv0, &tz0);

    for(int i = 1; i < m; i += 2) {
        int size = __builtin_popcount(i);
        mask[size].push_back(i);
    }

    dp[1][0] = 0;

#pragma omp parallel num_threads (nThreads)
    for(int size = 2; size <= n; size++) {
#pragma omp for
        for(int i = 0; i < mask[size].size(); i++) {
            int subset = mask[size][i];
            dp[subset][0] = INT_MAX;
            for(int i1 = 1; i1 < n; i1++) {
                int val = dp[subset][i1], pos;
                for(int i2 = 0; i2 < n; i2++) {
                    int p1 = 1 << i1, p2 = 1 << i2;
                    if((p1 & subset) == 0 || (p2 & subset) == 0 || i1 == i2) continue;
                    if(dp[subset ^ p1][i2] == INT_MAX) continue;
                    if(val > dp[subset ^ p1][i2] + d[i1][i2]) {
                        val = dp[subset ^ p1][i2] + d[i1][i2];
                        pos = i2;
                    }
                }
            dp[subset][i1] = val;
                    trans[subset][i1] = pos;
            }
	    }
    }

    int ans = INT_MAX, st = -1;
//#pragma omp parallel num_threads (nThreads)
    for(int i = 1; i < n; i++) {
        if(dp[m - 1][i] == INT_MAX) continue;
	    if(dp[m - 1][i] + d[i][0] < ans) {
//#pragma omp critical
            if(dp[m - 1][i] + d[i][0] < ans) {
                ans = dp[m - 1][i] + d[i][0];
	        st = i;
            }
	    }
    }
    int subset = m - 1;
    vector<int> path;
    path.push_back(1);
    while(st) {
        path.push_back(st + 1);
	int nxt = trans[subset][st];
	subset ^= 1 << st;
	st = nxt;

    }
    path.push_back(1);

    gettimeofday(&tv1, &tz1);

    for(auto i: path) fout << i << " ";
    fout << endl;
    fout << ans << endl;
    cout << "Number of vertices: " << n << "\n";
    cout << "Number of threads: " << nThreads << "\n";
    cout << "Time: " << (tv1.tv_sec-tv0.tv_sec)*1000000+(tv1.tv_usec-tv0.tv_usec) << " microseconds\n";
    fin.close();
    fout.close();
    return 0;
}
