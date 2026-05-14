# HPC Project — Parallel Matrix Multiplication using MPI + OpenMP
# Team Members

- @SamaMostafa6
- @
- @
  

## Overview
This project presents a **High Performance Computing (HPC)** implementation of large-scale matrix multiplication using:

- **Sequential Programming**
- **OpenMP (Shared Memory Parallelism)**
- **MPI (Distributed Memory Parallelism)**
- **Hybrid MPI + OpenMP Parallelism**

The project evaluates and compares the performance, scalability, and numerical correctness of each approach for large matrices of sizes:

- **N = 2000**
- **N = 4000**
- **N = 8000**

---

# Project Objectives

The main objectives of this project are:

1. Implement a **sequential baseline** for matrix multiplication.
2. Develop a parallel implementation using:
   - **OpenMP** for multi-threaded shared-memory execution.
   - **MPI** for distributed-memory execution across processes.
   - **Hybrid MPI + OpenMP** combining both paradigms.
3. Measure and analyze:
   - Execution time
   - Scalability
   - Speedup
   - Numerical accuracy
4. Compare the efficiency of each approach for large-scale computations.

---

# Technologies Used

- **C++**
- **OpenMP**
- **MPI (MS-MPI / OpenMPI compatible)**
- **High Performance Computing Concepts**

---

# Matrix Multiplication Formula

```math
C_{ij} = \sum_{k=1}^{N} A_{ik}B_{kj}
```

Where:

- `A` and `B` are input matrices
- `C` is the resulting matrix
- `N` is the matrix dimension

---

# Project Structure

```bash
HPC_project/
│
├── sequential_version.cpp
├── openmp_version.cpp
├── mpi_version.cpp
├── hybrid_version.cpp
│
├── README.md
└── results/
```

---

# Compilation Instructions

## 1. Sequential Version

Compile using:

```bash
g++ -O3 sequential_version.cpp -o sequential_run
```

---

## 2. OpenMP Version

Compile using:

```bash
g++ -O3 -fopenmp openmp_version.cpp -o openmp_run
```

---

## 3. MPI Version

### Linux / OpenMPI

```bash
mpic++ -O3 mpi_version.cpp -o mpi_run
```

### Windows (MS-MPI)

```bash
g++ mpi_version.cpp -o mpi_run.exe ^
-I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" ^
-L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" ^
-lmsmpi
```

---

## 4. Hybrid MPI + OpenMP Version

### Linux / OpenMPI

```bash
mpic++ -O3 -fopenmp hybrid_version.cpp -o hybrid_run
```

### Windows (MS-MPI)

```bash
g++ -O3 -fopenmp hybrid_version.cpp -o hybrid_run.exe ^
-I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" ^
-L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" ^
-lmsmpi
```

---

# Running the Programs

## Sequential Execution

```bash
./sequential_run 8000
```

---

## OpenMP Execution

Set the number of threads first.

### Linux/macOS

```bash
export OMP_NUM_THREADS=16
./openmp_run 8000
```

### Windows PowerShell

```powershell
$env:OMP_NUM_THREADS=16
.\openmp_run.exe 8000
```

---

## MPI Execution

Run using 8 MPI processes:

```bash
mpirun -np 8 ./mpi_run 8000
```

---

## Hybrid MPI + OpenMP Execution

Example: **4 MPI processes × 4 OpenMP threads**

### Linux/macOS

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 ./hybrid_run 8000
```

### Windows PowerShell

```powershell
$env:OMP_NUM_THREADS=4
mpirun -np 4 .\hybrid_run.exe 8000
```

---

# Performance Metrics

The following metrics are evaluated:

- Execution Time
- Parallel Speedup
- Efficiency
- Scalability
- CPU Utilization
- Numerical Accuracy

---

# Parallel Programming Models

## OpenMP
OpenMP uses **multi-threading** for shared-memory systems where threads run concurrently on multiple CPU cores.

## MPI
MPI distributes computations across multiple processes, enabling execution on distributed-memory systems and clusters.

## Hybrid MPI + OpenMP
The hybrid model combines:
- MPI between nodes/processes
- OpenMP within each process

This approach improves scalability and resource utilization for large HPC workloads.

---

# Expected Results

The hybrid implementation is expected to provide:

- Better scalability for very large matrices
- Reduced execution time
- Improved CPU utilization
- Higher performance compared to purely sequential execution

---

# Authors

High Performance Computing Course Project  
Hybrid Matrix Multiplication using MPI + OpenMP
