#include <mpi.h>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
// #define CHECK

void generate_system(std::vector<double>& matrix, std::vector<double>& b, int N) {
    srand(time(NULL));  
    int scale = 100;  

    matrix.resize(N * N);
    b.resize(N);

    for (int i = 0; i < N; ++i) {
        double rowSum = 0; 
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                matrix[i * N + j] = (double)(rand() % scale) - scale / 2.0; 
                rowSum += std::abs(matrix[i * N + j]);
            }
        }
        matrix[i * N + i] = rowSum + 1; 
        b[i] = rand() % (scale * N) - (scale * N) / 2.0; 
    }
}

#ifdef CHECK
void gauss_elimination(std::vector<double>& A, std::vector<double>& b, std::vector<double>& x, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = i + 1; k < N; k++) {
            double factor = -A[k * N + i] / A[i * N + i];
            for (int j = i; j < N; j++) {
                if (i == j) {
                    A[k * N + j] = 0;
                } else {
                    A[k * N + j] += factor * A[i * N + j];
                }
            }
            b[k] += factor * b[i];
        }
    }

    for (int i = N - 1; i >= 0; i--) {
        double sum = b[i];
        for (int j = i + 1; j < N; j++) {
            sum -= A[i * N + j] * x[j];
        }
        x[i] = sum / A[i * N + i];
    }
}
#endif

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Incorrect function call. You should run mpirun -np 5 ./FourthTask N" << std::endl;
        return 1;
    }
    int N = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    double startTime, endTime;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> A(N * N);
    std::vector<double> b(N);
    std::vector<double> x(N);

    if (rank == 0) {
        A.resize(N * N);
        b.resize(N);
        x.resize(N);
        generate_system(A, b, N);
        startTime = MPI_Wtime();
    }
    
    int sizeForProc = N / size;
    int remainder = N % size;
    int localSize = ((rank < remainder) ? (sizeForProc + 1) : sizeForProc) * N;
    std::vector<double> localA(localSize);
    std::vector<double> localb(localSize / N);

    std::vector<int> localSizes(size), offsets(size);
    std::vector<int> localSizesb(size), offsetsb(size);
    int offset = 0;
    int offsetb = 0;
    for (int i = 0; i < size; i++) {
        localSizes[i] = ((i < remainder) ? (sizeForProc + 1) : sizeForProc) * N;
        offsets[i] = offset;
        offset += localSizes[i];

        localSizesb[i] = (i < remainder) ? sizeForProc + 1 : sizeForProc;
        offsetsb[i] = offsetb;
        offsetb += localSizesb[i];
    }

    MPI_Scatterv(A.data(), localSizes.data(), offsets.data(), MPI_DOUBLE, localA.data(), localSizes[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b.data(), localSizesb.data(), offsetsb.data(), MPI_DOUBLE, localb.data(), localSizesb[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N - 1; ++i) {
        int rowOwner = i / sizeForProc;
        double referenceElement = 0;
        if (rank == rowOwner) {
            referenceElement = localA[(i % sizeForProc) * N + i];
        }
        MPI_Bcast(&referenceElement, 1, MPI_DOUBLE, rowOwner, MPI_COMM_WORLD);

        for (int j = ((rank > rowOwner) ? 0 : (i % sizeForProc) + 1); j < sizeForProc; ++j) {
            double factor = localA[j * N + i] / referenceElement;
            for (int k = i; k < N; ++k) {
                localA[j * N + k] -= factor * localA[(i % sizeForProc) * N + k];
            }
            localb[j] -= factor * localb[i % sizeForProc];
        }
    }

    MPI_Gatherv(localA.data(), localSizes[rank], MPI_DOUBLE, A.data(), localSizes.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localb.data(), localSizesb[rank], MPI_DOUBLE, b.data(), localSizesb.data(), offsetsb.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = N - 1; i >= 0; --i) {
            double sum = b[i];
            for (int j = i + 1; j < N; ++j) {
                sum -= A[i * N + j] * x[j];
            }
            x[i] = sum / A[i * N + i];
        }
    }

    if (rank == 0) {
        endTime = MPI_Wtime();
        #ifdef CHECK
        std::vector<double> xCheck(N);
        auto start = std::chrono::high_resolution_clock::now();
        gauss_elimination(A, b, xCheck, N);
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = finish - start;
        int max = -1;
        for (int i = 0; i < N; ++i)
        {
            if(fabs(x[i]-xCheck[i])) max = fabs(x[i]-xCheck[i]);
        }
        std::cout << "1 thread. N = " << N <<" Time taken: " << elapsed.count()*1000 << " ms." << std::endl;
        std::cout << "error = " << max << std::endl;
        #endif

        std::cout << "N = " << N <<" Time taken: " << (endTime - startTime) * 1000 << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}