#include "knnring.h"
#include "mpi.h"
#include "string.h"

void mergeResults(knnresult* A, knnresult* B);

knnresult distrAllkNN(double* X, int n, int d, int k)
{
	int nProcs, myRank, myNext, myPrev;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	myPrev = myRank==0 ? nProcs-1 : myRank-1;
	myNext = (myRank+1)%nProcs; 

	// For now, each process takes care of it's own small query set using the sequential version!
	// Eventually process 0 will collect the results and compile the final answer.
	double* Y = (double*) malloc(n*d*sizeof(double));
	memcpy(Y, X, n*d*sizeof(double));
	knnresult result = knn(X, Y, n, n, d, k);
	for (int i=0; i<nProcs-1; i++) //we need to transfer data in the ring nProcs-1 times
	{
		MPI_Sendrecv_replace(Y, n*d, MPI_Double, myNext, MPI_ANY_TAG, myPrev, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		knnresult result2 = knn(X, Y, n, n, d, k);
		mergeResultsAndClear(&result, &result2);
	}
	
	//Now it's time to gather the 
}

void mergeResults(knnresult* A, knnresult* B)
{
	int* nidx 		= (int*) 	malloc(A->m*A->k*sizeof(int));
	double* ndist	= (double*)	malloc(A->m*A->k*sizeof(double));
	int i=0, j=0;
	for (int neighbor=0; neighbor<k; neighbor++)
		if (A->ndist[i] < B->ndist[j])
		{
			nidx [neighbor] = A->nidx[i];
			ndist[neighbor] = A->ndist[i];
			i++;
		}
		else 
		{
			nidx [neighbor] = B->nidx[j];
			ndist[neighbor] = B->ndist[j];
			j++;
		}
	free(A->nidx);
	free(A->ndist);
	free(B->nidx);
	free(B->ndist);
	A->nidx  = nidx;
	A->ndist = ndist;
}

