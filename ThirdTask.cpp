#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
//#define CHECK


double f(double x, double y) {
    return cos(x*x) * sin(y);
}

#ifdef CHECK
double g(double x, double y) {
    return cos(x*x) * cos(y);
}
#endif

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Incorrect function call. You should run mpirun -np 5 ./ThirdTask N M" << std::endl;
        return 1;
    }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    MPI_Init(&argc, &argv);
    double startTime, endTime;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double xStart = 1, xEnd = 10;
    double yStart = 1, yEnd = 10;
    
    double hx = (xEnd - xStart)/(double)N;
    double hy = (yEnd - yStart)/(double)M;
   
    std::vector<double> Function;
    std::vector<double> Derivative;

    if (rank == 0) {
        Function.resize(N * M);
        Derivative.resize(N * M, 0.0);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                Function[i * M + j] = f(xStart + hx * i, yStart + hy * j);
            }
        }
        startTime = MPI_Wtime();
    }

    int sizeForProc = N / size;
    int remainder = N % size;
    int localSize = ((rank < remainder) ? (sizeForProc + 1) : sizeForProc) * M;
    std::vector<double> localFun(localSize);
    std::vector<double> localDerivative(localSize, 0.0);

    std::vector<int> localSizes(size), offsets(size);
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        localSizes[i] = ((i < remainder) ? (sizeForProc + 1) : sizeForProc) * M;
        offsets[i] = offset;
        offset += localSizes[i];
    }

    MPI_Scatterv(rank == 0 ? Function.data() : nullptr, localSizes.data(), offsets.data(), MPI_DOUBLE,
                 localFun.data(), localSize, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    for (int i = 0; i < localSize / M; ++i) {
        for (int j = 0; j < M; ++j) {
            double fm = (j == 0) ? localFun[i * M + j] : localFun[i * M + (j - 1)];
            double fp = (j == M - 1) ? localFun[i * M + j] : localFun[i * M + (j + 1)];
            double divider = (j == 0 || j == M - 1) ? hy : 2 * hy;
            localDerivative[i * M + j] = (fp - fm) / divider;
        }
    }

    MPI_Gatherv(localDerivative.data(), localSize, MPI_DOUBLE,
                rank == 0 ? Derivative.data() : nullptr, localSizes.data(), offsets.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        endTime = MPI_Wtime();
        #ifdef CHECK
        double max = -1;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                if (fabs(Derivative[i * M + j] - g(xStart + hx * i, yStart + hy * j)) > max)
                    max = fabs(Derivative[i * M + j] - g(xStart + hx * i, yStart + hy * j));
            }
        }
        std::cout << "error = " << max << std::endl;
        #endif
        std::cout << "N = " << N << " M = " << M <<" Time taken: " << (endTime - startTime) * 1000 << " ms" << std::endl;
    }

    MPI_Finalize();

    return 0;
}