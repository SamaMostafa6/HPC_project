# HPC_project
"Hybrid MPI + OpenMP implementation and performance analysis of large-scale matrix multiplication for N=2000, 4000, and 8000."

#----------------------------------

High Performance Computing - Matrix Multiplication
This project implements parallel matrix multiplication using OpenMP, MPI, and a Hybrid approach.  

Project Goals:
1.Implement sequential baseline multiplication.  

2.Develop parallel versions using OpenMP (Shared Memory) and MPI (Distributed Memory).  

3.Analyze scalability and numerical accuracy for large matrices (up to 8000x8000).  

How to Compile
Use g++ for OpenMP and mpic++ for MPI-based versions:

Sequential: g++ -O3 -o sequential_version2 sequential_version.cpp

OpenMP: g++ -O3 -fopenmp openmp_version.cpp -o openmp_version.exe

MPI: g++ mpi_version.cpp -o mpi_version.exe `
-I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" `
-L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" `
-lmsmpi

Hybrid: g++ -O3 -fopenmp hybird_version.cpp -o hybrid_mm.exe -I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
$env:OMP_NUM_THREADS=4 //to tell the openmp the threads number

How to Run
Sequential: ./sequential_run 8000

OpenMP (16 threads): export OMP_NUM_THREADS=16 && ./openmp_run 8000

MPI (8 processes): mpirun -np 8 ./mpi_run 8000

Hybrid (4 procs x 4 threads): mpirun -np 4 ./hybrid_run 8000 4
