#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace chrono;

void matrix_multiply_sequential(const vector<double>& A,
                                 const vector<double>& B,
                                 vector<double>& C,
                                 int N) {
    fill(C.begin(), C.end(), 0.0);

    int progress_step = N / 10; // update every 10%
    if (progress_step == 0) progress_step = 1;

    for (int i = 0; i < N; i++) {

        //  Progress logging
        if (i % progress_step == 0) {
            double percent = (100.0 * i) / N;
            cout << "\rProgress: " << fixed << setprecision(1)
                 << percent << "%" << flush;
        }

        for (int k = 0; k < N; k++) {
            double a_val = A[i * N + k];
            double* c_row = &C[i * N];
            const double* b_row = &B[k * N];

            for (int j = 0; j < N; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }

    cout << "\rProgress: 100.0%" << endl;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 8000;
    
    cout << "\n========== SEQUENTIAL MATRIX MULTIPLICATION ==========" << endl;
    cout << "Matrix Size: " << N << " x " << N << endl;
    cout << "Memory: " << (3LL * N * N * sizeof(double)) / (1024 * 1024) << " MB" << endl;
    cout << "=======================================================" << endl;
    
    vector<double> A(N * N), B(N * N), C(N * N);
    
    random_device rd;
    mt19937 gen(42); // Fixed seed for reproducibility
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    for (int i = 0; i < N * N; i++) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }
    
    cout << "Running benchmark..." << endl;
    auto start = high_resolution_clock::now();
    
    matrix_multiply_sequential(A, B, C, N);
    
    auto end = high_resolution_clock::now();
    double seconds = duration_cast<milliseconds>(end - start).count() / 1000.0;
    
    long long ops = 2LL * N * N * N;
    double gflops = ops / (seconds * 1e9);
    
    cout << fixed << setprecision(4);
    cout << "\n--- RESULTS ---" << endl;
    cout << "Time: " << seconds << " seconds" << endl;
    cout << "GFLOPS: " << gflops << endl;
    
    // Verification sample
    cout << "\nSample C[0][0]: " << C[0] << endl;
    
    return 0;
}