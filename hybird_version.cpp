#include <iostream>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// Helper for clean logging
#define LOG(rank, msg) cout << "[Rank " << rank << "] " << msg << endl

void hybrid_multiply(const double* A, const double* B, double* C,
                     int N, int local_rows, int num_threads, int rank) {
    
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for
    for (int i = 0; i < local_rows; i++) {
        // Progress reporting: Only Rank 0, Thread 0 handles the console update
        if (rank == 0 && omp_get_thread_num() == 0 && i % max(1, (local_rows / 20)) == 0) {
            float progress = (float)i / local_rows * 100;
            #pragma omp critical
            {
                cout << "\r[Rank 0] Computing: " << fixed << setprecision(1) << progress << "%" << flush;
            }
        }

        // Cache-friendly i-k-j loop order
        for (int k = 0; k < N; k++) {
            double a_val = A[i * N + k];
            #pragma omp simd
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
    if (rank == 0) cout << "\r[Rank 0] Computing: 100.0% - Done!          " << endl;
}

int main(int argc, char* argv[]) {
    // 1. Initialize MPI with thread safety
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Default parameters if not provided via command line
    int N = (argc > 1) ? atoi(argv[1]) : 2000; 
    int threads_per_proc = (argc > 2) ? atoi(argv[2]) : 4;
    
    if (rank == 0) {
        cout << "========================================" << endl;
        cout << "   HYBRID MPI + OPENMP MATRIX MULT      " << endl;
        cout << "========================================" << endl;
        cout << "Matrix Size: " << N << " x " << N << endl;
        cout << "MPI Tasks:   " << size << endl;
        cout << "Threads/Task: " << threads_per_proc << endl;
    }

    // 2. Row distribution logic
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    
    vector<int> send_counts(size), displacements(size);
    int offset = 0;
    for (int i = 0; i < size; i++) {
        send_counts[i] = ((i < remainder) ? rows_per_proc + 1 : rows_per_proc) * N;
        displacements[i] = offset;
        offset += send_counts[i];
    }
    
    // 3. Memory Allocation with basic error checking
    double *local_A = new (nothrow) double[local_rows * N];
    double *local_C = new (nothrow) double[local_rows * N]();
    double *A = nullptr, *B = nullptr, *C = nullptr;
    
    if (!local_A || !local_C) {
        LOG(rank, "FATAL: Local memory allocation failed!");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A = new (nothrow) double[N * N];
        B = new (nothrow) double[N * N];
        C = new (nothrow) double[N * N]();
        if (!A || !B || !C) {
            LOG(rank, "FATAL: Global memory allocation failed!");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Initialize with random data
        for (int i = 0; i < N * N; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
    } else {
        // Workers need to allocate B to receive the broadcast
        B = new (nothrow) double[N * N];
        if (!B) {
            LOG(rank, "FATAL: Memory for matrix B failed!");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // 4. Communication
    MPI_Scatterv(A, send_counts.data(), displacements.data(), MPI_DOUBLE,
                 local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // 5. Execution & Timing
    MPI_Barrier(MPI_COMM_WORLD); 
    double start_time = MPI_Wtime();
    
    hybrid_multiply(local_A, B, local_C, N, local_rows, threads_per_proc, rank);
    
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;
    
    // 6. Gather results back to Rank 0
    MPI_Gatherv(local_C, local_rows * N, MPI_DOUBLE,
                C, send_counts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    
    // 7. Final Report
    double max_time = 0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "\n---------- PERFORMANCE REPORT ----------" << endl;
        cout << "Execution Time: " << fixed << setprecision(4) << max_time << " s" << endl;
        if (max_time > 0) {
            long long ops = 2LL * N * N * N;
            double gflops = ops / (max_time * 1e9);
            cout << "Throughput:     " << gflops << " GFLOPS" << endl;
        }
        cout << "----------------------------------------" << endl;
        
        delete[] A; delete[] B; delete[] C;
    } else {
        delete[] B;
    }
    
    delete[] local_A; delete[] local_C;
    
    MPI_Finalize();
    return 0;
}