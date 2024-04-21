#include <mpi.h>
#include <iostream>
//#define CHECK

int* createRandomArrayNumbers(int size) {
    int* array = new int[size];
    for (int i = 0; i < size; ++i) {
        array[i] = i;
    }
    return array;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Incorrect function call. You should run mpirun -np 5 ./SeconTask arraySize" << std::endl;
        return 1;
    }
    int arraySize = atoi(argv[1]); 
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime, endTime;

    int* data = nullptr;

    if (rank == 0) {
        data = createRandomArrayNumbers(arraySize);
        startTime = MPI_Wtime();
    }

    int localArraySize = arraySize / size;
    int* localData = new int[localArraySize];
    MPI_Scatter(data, localArraySize, MPI_INT, localData, localArraySize, MPI_INT, 0, MPI_COMM_WORLD);

    int localSum = 0;
    for (int j = 0; j < localArraySize; ++j) {
        localSum += localData[j];
    }

    long totalSum = 0;
    MPI_Reduce(&localSum, &totalSum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        endTime = MPI_Wtime();
        #ifdef CHECK
        std::cout << "Total sum for array of size " << n << " is " << totalSum << std::endl;
        #endif
        std::cout << "arraySize = " << arraySize <<" Time taken: " << (endTime - startTime) * 1000 << " ms" << std::endl;
        delete[] data; 
    }

    delete[] localData; 

    MPI_Finalize();
    return 0;
}