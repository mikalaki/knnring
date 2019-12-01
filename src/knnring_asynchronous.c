#include "knnring.h"
#include "mpi.h"
#include "string.h"
#include "stdlib.h"
#include "stdio.h"

void mergeResultsAndClear(knnresult* A, knnresult* B);
inline void addToIndexes(knnresult* R, int addVal);

knnresult distrAllkNN(double* X, int n, int d, int k)
{
	int nProcs, myRank, myNext, myPrev;

	MPI_Comm_rank(MPI_COMM_WORLD, &myRank); //get the process ID (rank)
	MPI_Comm_size(MPI_COMM_WORLD, &nProcs); //get the number of procceses

	//Setting the source and the destination for each procces.
	myPrev = myRank==0 ? nProcs-1 : myRank-1;
	myNext = (myRank+1)%nProcs;

	/* Y will be the query set. This will be fixed for every process.
	 * corpus[0] and corpus[1] are the corpus sets that are being exchanged.
	 * The sets are used alternateviy; One of them is being sent and at
	 * the same time used by kNN, while the other is used for receiving data
	 * from the previous process. For this, we use the modulo 2 trick.
	 */
	double* Y = (double*) malloc(n*d*sizeof(double));
	memcpy(Y, X, n*d*sizeof(double));
	double* corpus[2] = {X, NULL};
	corpus[1] = (double*) malloc(n*d*sizeof(double));

	// Start sending and receiving at corpus[1] before calling kNN for corpus[0]
	MPI_Request requests[2];
	MPI_Isend(corpus[0], n*d, MPI_DOUBLE, myNext, 2019, MPI_COMM_WORLD, &requests[0]);
	MPI_Irecv(corpus[1], n*d, MPI_DOUBLE, myPrev, MPI_ANY_TAG, MPI_COMM_WORLD, &requests[1]);

	/* Each process takes care of it's own small query set using the sequential version!
	 * The sequential vesion assumes that the indexing of the points starts from 0
	 * This is true only for process 0, and thus correct for sequential implementation.
	 * To fix, we add (myRank-1)*n to every index. This way, we also assume that each
	 * process parses equal amount of points, which holds true for the given tester.
	 * Of course, an implementation that doesn't make the above assumption is possible,
	 * but harder due to the current interface and not part of the assignment.
	 */
	knnresult result = kNN(X, Y, n, n, d, k);
	addToIndexes(&result, n*(myRank>0 ? myRank-1 : nProcs-1) );

	//We need to transfer data in the ring nProcs-1 times
	for (int i=1; i<nProcs; i++)
	{
		MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
		MPI_Isend(corpus[	 i%2], n*d, MPI_DOUBLE, myNext, 2019, MPI_COMM_WORLD, &requests[0]);
		MPI_Irecv(corpus[(i+1)%2], n*d, MPI_DOUBLE, myPrev, MPI_ANY_TAG, MPI_COMM_WORLD, &requests[1]);

		knnresult result2 = kNN(corpus[i%2], Y, n, n, d, k);
		addToIndexes(&result2, n*(myRank-i-1>=0 ? myRank-i-1 : myRank-i-1+nProcs));
		mergeResultsAndClear(&result, &result2);
	}

	//The result is ready, compute global min and max before returning
	double minDist=result.ndist[0*k +0];
	double maxDist=result.ndist[0*k +k-1];
	double globalMin, globalMax;

	for (int i=1; i<n; i++)
	{
		if (result.ndist[i*k +0] < minDist) minDist = result.ndist[i*k +0];
		if (result.ndist[i*k +k-1] > maxDist) maxDist = result.ndist[i*k +k-1];
	}

	MPI_Reduce(&minDist, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&maxDist, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (myRank==0)
		printf("This is root (proc num 0) speaking:\n--->Global distances minimum=%lf, maximum=%lf\n",globalMin,globalMax);

	return result;
}

void mergeResultsAndClear(knnresult* A, knnresult* B)
{
	int k=A->k, m=A->m;
	int* nidx 		= (int*) 	malloc(m*k*sizeof(int));
	double* ndist	= (double*)	malloc(m*k*sizeof(double));

	//Scanning the two matricies and getting the smallest k distances
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
	//Deallocating memory
	free(A->nidx);
	free(A->ndist);
	free(B->nidx);
	free(B->ndist);

	A->nidx  = nidx;
	A->ndist = ndist;
}

//Function to add offset to the idx array depending on which part of CorpusAll come from
inline void addToIndexes(knnresult* R, int addVal)
{
	for (int i=0; i<R->m*R->k; i++)
		R->nidx[i] += addVal;
}
