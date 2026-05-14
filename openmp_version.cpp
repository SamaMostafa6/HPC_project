#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <omp.h>

using namespace std;
using namespace chrono;

void matrix_multiply_openmp(const vector<double>& A,
                            const vector<double>& B,
                            vector<double>& C,
                            int N) {
    fill(C.begin(), C.end(), 0.0);
    
    // Project Requirement: Parallel for with collapse(2)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            double a_val = A[i * N + k];
            // SIMD optimization for the innermost loop
            #pragma omp simd
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 8000;
    int num_threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    
    // Your actual baseline from the sequential run
    double sequential_baseline_2000 = 31.7020; 
    
    omp_set_num_threads(num_threads);
    
    cout << "\n========== OPENMP MATRIX MULTIPLICATION ==========" << endl;
    cout << "Matrix Size: " << N << " x " << N << endl;
    cout << "Threads: " << num_threads << endl;
    cout << "===================================================" << endl;
    
    vector<double> A(N * N), B(N * N), C(N * N);
    
    // Fixed seed for accuracy comparison requirement
    mt19937 gen(42); 
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < N * N; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
    
    auto start = high_resolution_clock::now();
    matrix_multiply_openmp(A, B, C, N);
    auto end = high_resolution_clock::now();
    
    double seconds = duration_cast<milliseconds>(end - start).count() / 1000.0;
    long long ops = 2LL * N * N * N;
    double gflops = ops / (seconds * 1e9);
    
    cout << fixed << setprecision(4);
    cout << "\n--- RESULTS ---" << endl;
    cout << "Time: " << seconds << " seconds" << endl;
    cout << "GFLOPS: " << gflops << endl;
    
    if (N == 2000) {
        cout << "Speedup: " << (sequential_baseline_2000 / seconds) << "x" << endl;
    }
    
    cout << "Sample C[0][0]: " << C[0] << " (Check against 26.7217)" << endl;
    
    return 0;
}