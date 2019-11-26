#include "knnring.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"

void mergeResultsAndClear(knnresult* A, knnresult* B);
inline void addToIndexes(knnresult* R, int addVal);

//~ void print(knnresult* R, FILE* file)
//~ {
	//~ fprintf(file,"%d %d\n", R->m, R->k);
	//~ for (int i=0; i<R->m; i++)
	//~ {
		//~ for (int j=0; j<R->k; j++) fprintf(file,"%d ",R->nidx[i*R->k +j]);
		//~ fprintf(file,"\n");
	//~ }
	//~ fprintf(file,"\n\n");
	//~ fflush(file);
//~ }

knnresult distrAllkNN(double* X, int n, int d, int k)
{
	int nProcs, myRank, myNext, myPrev;
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
	myPrev = myRank==0 ? nProcs-1 : myRank-1;
	myNext = (myRank+1)%nProcs; 
	
	//~ char filename[100];
	//~ sprintf(filename,"proc%d",myRank);
	//~ FILE* file = fopen(filename,"w");

	/* Each process takes care of it's own small query set using the sequential version!
	 * The sequential vesion assumes that the indexing of the points starts from 0
	 * This is true only for process 0, and thus correct for sequential implementation.
	 * To fix, we add myRank*result.m to every index. This way, we also assume that each 
	 * process parses equal amount of points, which holds true for the given tester.
	 * Of course, an implementation that doesn't make the above assumption is possible,
	 * but harder due to the current interface and not part of the assignment.
	 */ 
	double* Y = (double*) malloc(n*d*sizeof(double));
	memcpy(Y, X, n*d*sizeof(double));
	knnresult result = kNN(X, Y, n, n, d, k);
	addToIndexes(&result, myRank*n);
	//~ print(&result, file);
	
	//We need to transfer data in the ring nProcs-1 times
	for (int i=1; i<nProcs; i++) 
	{
		MPI_Sendrecv_replace(Y, n*d, MPI_DOUBLE, myNext, 2019, myPrev, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		knnresult result2 = kNN(X, Y, n, n, d, k);
		addToIndexes(&result2, n*(myRank-i>=0 ? myRank-i : myRank-i+nProcs)); 
		//~ print(&result2, file);
		mergeResultsAndClear(&result, &result2);
		//~ print(&result, file);
	}
	
	return result;
}

void mergeResultsAndClear(knnresult* A, knnresult* B)
{
	int k=A->k, m=A->m;
	int* nidx 		= (int*) 	malloc(m*k*sizeof(int));
	double* ndist	= (double*)	malloc(m*k*sizeof(double));
	
	for (int point=0; point<m; point++)
		for (int i=0,j=0, neighbor=0; neighbor<k; neighbor++)
			if (A->ndist[point*k +i] < B->ndist[point*k +j])
			{
				nidx [point*k +neighbor] = A->nidx[point*k +i];
				ndist[point*k +neighbor] = A->ndist[point*k +i];
				i++;
			}
			else 
			{
				nidx [point*k +neighbor] = B->nidx[point*k +j];
				ndist[point*k +neighbor] = B->ndist[point*k +j];
				j++;
			}
	free(A->nidx);
	free(A->ndist);
	free(B->nidx);
	free(B->ndist);
	A->nidx  = nidx;
	A->ndist = ndist;
}

inline void addToIndexes(knnresult* R, int addVal)
{
	for (int i=0; i<R->m*R->k; i++)
		R->nidx[i] += addVal;
}
