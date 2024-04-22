#include <mpi.h>
#include <vector>
#include <iostream>
#include <cmath>
// #define CHECK

void generateMatrix(std::vector<double>& matrix, int N) {
    int rand = time(NULL)%17;
    int i = 1;
    for (auto& element : matrix) {
        element = rand * i / (double)(13);
        ++i;
    }
}

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

    std::vector<double> A;
    std::vector<double> B(N * N);
    std::vector<double> C;

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N, 0.0);
        generateMatrix(A, N);
        generateMatrix(B, N);
        startTime = MPI_Wtime();
    }

    int sizeForProc = N / size;
    int remainder = N % size;
    int localSize = ((rank < remainder) ? (sizeForProc + 1) : sizeForProc) * N;
    std::vector<double> localA(localSize);
    std::vector<double> localC(localSize, 0.0);

    std::vector<int> localSizes(size), offsets(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        localSizes[i] = ((i < remainder) ? (sizeForProc + 1) : sizeForProc) * N;
        offsets[i] = offset;
        offset += localSizes[i];
    }

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, localSizes.data(), offsets.data(), MPI_DOUBLE,
                 localA.data(), localSize, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Bcast(B.data(), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < localSize / N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += localA[i * N + k] * B[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }

    MPI_Gatherv(localC.data(), localSize, MPI_DOUBLE,
            rank == 0 ? C.data() : nullptr, localSizes.data(), offsets.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD);


    if (rank == 0) {
        endTime = MPI_Wtime();
        #ifdef CHECK
        double max = -1;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                if (fabs(C[i * N + j] - sum) > max)
                        max = fabs(C[i * N + j] - sum);
            }
        }
        std::cout << "error = " << max << std::endl;
        #endif
        std::cout << "N = " << N <<" Time taken: " << (endTime - startTime) * 1000 << " ms" << std::endl;
    }

    MPI_Finalize();
    return 0;
}