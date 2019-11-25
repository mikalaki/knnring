#include "knnring.h"
#include "mpi.h"

int myRank;

knnresult distrAllkNN(double* X, int n, int d, int k)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	double* distArr
}
