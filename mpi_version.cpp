#include <iostream>
#include <vector>
#include <mpi.h>
#include <iomanip>
#include <cstdlib>

using namespace std;

// logging logic
void matrix_multiply_block(const double* A, const double* B, double* C, int N, int local_rows, int rank) {
    int progress_step = local_rows / 10; // Update every 10% of local work
    if (progress_step == 0) progress_step = 1;

    for (int i = 0; i < local_rows; i++) {
        
        // Progress logging: Only Rank 0 prints to avoid terminal clutter
        if (rank == 0 && i % progress_step == 0) {
            double percent = (100.0 * i) / local_rows;
            cout << "\rProgress: " << fixed << setprecision(1) << percent << "%" << flush;
        }

        for (int k = 0; k < N; k++) {
            double a_val = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
    
    // all the work done
    if (rank == 0) {
        cout << "\rProgress: 100.0%" << endl;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = (argc > 1) ? atoi(argv[1]) : 8000;
    
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

    double *A = nullptr, *B = nullptr, *C = nullptr;
    double *local_A = new double[local_rows * N];
    double *local_C = new double[local_rows * N]();

    if (rank == 0) {
        cout << fixed << setprecision(2);
        cout << "\n========== MPI ==========" << endl;
        cout << "Matrix Size: " << N << " x " << N << endl;
        cout << "Memory:      " << (3LL * N * N * sizeof(double)) / (1024 * 1024) << " MB" << endl;
        cout << "Processes:   " << size << endl;
        cout << "=======================================================" << endl;

        A = new double[(size_t)N * N];
        B = new double[(size_t)N * N];
        C = new double[(size_t)N * N]();
        
        srand(42);
        for (size_t i = 0; i < (size_t)N * N; i++) {
            A[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            B[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    // Broadcast B and Scatter A to all processes
    if (rank != 0) B = new double[(size_t)N * N];
    MPI_Bcast(B, (size_t)N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(A, send_counts.data(), displacements.data(), MPI_DOUBLE, 
                 local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    
    // Computation starts here
    matrix_multiply_block(local_A, B, local_C, N, local_rows, rank);
    
    double end_time = MPI_Wtime();
    double local_time = end_time - start_time;

    // Gather results back to Root
    MPI_Gatherv(local_C, local_rows * N, MPI_DOUBLE, 
                C, send_counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << fixed << setprecision(4);
        cout << "\n========== MPI PERFORMANCE RESULTS ==========" << endl;
        cout << "Total Time:  " << max_time << " seconds" << endl;
        long long ops = 2LL * N * N * N;
        cout << "GFLOPS:      " << ops / (max_time * 1e9) << endl;
        cout << "Sample C[0]: " << C[0] << " (Check: 26.7217)" << endl;
        cout << "=============================================" << endl;
        
        delete[] A; delete[] B; delete[] C;
    }

    delete[] local_A; delete[] local_C;
    if (rank != 0) delete[] B;
    
    MPI_Finalize();
    return 0;
}