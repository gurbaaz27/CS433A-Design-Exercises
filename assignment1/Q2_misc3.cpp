#include<bits/stdc++.h>
#include<omp.h>
#include <sys/time.h>

using namespace std;

// Compute y given L and x
void compute_y(int n, long double** L, long double* x, long double* y){
#pragma omp parallel for num_threads(8)
    for(int i=0; i<n; ++i){
        y[i] = 0;
        for(int j=0; j<=i; ++j){
            y[i] += x[j]*L[i][j];
        }
    }
}

// Initialize x and L to arbitrary values and compute y using them
void initialize_input(int n, long double** L, long double* x, long double* y, pair<long double, long double> range = {-1,1}){
    srand(time(0));
    for(int row=0; row<n; ++row){
        x[row] = (static_cast<long double>(rand())/RAND_MAX)*(range.second - range.first) + range.first;
        for(int col=0; col<=row; ++col){
            L[row][col] =
                    (static_cast<long double>(rand())/RAND_MAX)*(range.second - range.first) + range.first;
        }
    }
    compute_y(n, L, x, y);
}

// Check for equality
bool are_equal(long double a, long double b){
    long double diff = fabs(a-b);
    return diff < 0.01;
}

// Verify the result
bool verify_x(int n, long double* x1, long double* x2){
    for(int i=0; i<n; ++i){
        if(!are_equal(x1[i], x2[i])){
            cout << i << " " << x1[i] << " " << x2[i] << endl;
            return false;
        }
    }
    return true;
}

// Solve for x using forward substitution
void compute_x(int n, long double** L, long double* x, long double* y){
// #pragma omp parallel for
    // for(int i=0; i<n; ++i){
    //     x[i] = y[i];
    // }
    
    for(int i=0; i<n; ++i){
        x[i] = y[i]/L[i][i];
// #pragma omp parallel for
        for(int j=i+1; j<n; ++j){
            y[j] -= L[j][i]*x[i];
        }
    }

//     for(int i=0; i<n; ++i){ 
// // #pragma omp parallel for
//         for(int j=0; j<i; ++j){
//             x[j] -= L[i][j]*x[j];
//         }
//         x[i] = y[i]/L[i][i];
//     }
}

void read_inputfile(fstream& fin, int n, long double **L, long double *y){
    for(int i=0; i<n; ++i){
        for(int j=0; j<=i; ++j){
            fin >> L[i][j];
        }
    }
    
    for(int i=0; i<n; ++i){
        fin >> y[i];
    }
}

// Print an array
void print(fstream& fout, int n, long double* arr){
    for(int i=0; i<n; ++i){
        fout << arr[i] << "\n";
    }
}

long double** L;
long double* y;
long double *true_x;
long double* x;
int thread_count;
int n;

int main(int argc, char* argv[]){
    // Processing arguments
    if(argc != 4){
        cout << "Invalid no. of arguments! Correct Usage:\n\n";
        cout << argv[0] << " <input file> <output file> <number of threads>";
    }

    thread_count = stoi(argv[3]);
    
    fstream fin, fout;
    try
    {
        fin.open(argv[1], ios_base::in);
        fout.open(argv[2], ios_base::out);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception occcured while opening input/output file : " << e.what() << '\n';
    }

    // Initializing n
    // n = 10000;
    fin >> n;

    // Allocating Memory
    L = (long double**)malloc(n*__SIZEOF_POINTER__);
    for(int i=0; i<n; ++i){
        L[i] = (long double*)malloc((i+1)*__SIZEOF_LONG_DOUBLE__);
    }
    y = (long double*)malloc(n*__SIZEOF_LONG_DOUBLE__);
    x = (long double*)malloc(n*__SIZEOF_LONG_DOUBLE__);
    true_x = (long double*)malloc(n*__SIZEOF_LONG_DOUBLE__);
    
    // Initialization
    read_inputfile(fin, n, L, y);
    // for(int i=0;i<n;i++){
    //     for(int j=0;j<n;j++){
    //         cout << L[i][j] << " ";
    //     }
    //     cout << "\n";
    // }
    // for(int i=0;i<n;i++)
    // cout <<y[i] << " ";
    // initialize_input(n, L, true_x, y, {1,5});
    
    struct timeval tv0, tv1;
    struct timezone tz0, tz1;

    // Computing x
    omp_set_num_threads(thread_count);
    gettimeofday(&tv0, &tz0);
    compute_x(n, L, x, y);
    gettimeofday(&tv1, &tz1);

    print(fout, n, x);

    cout << "n = " << n << ", num_th = " << thread_count << ", Time : " << 
                    (((tv1.tv_sec-tv0.tv_sec)*1000000.0+(tv1.tv_usec-tv0.tv_usec))/1000)<< " ms" << endl;

    // cout << boolalpha << verify_x(n, x, true_x) << endl;
}