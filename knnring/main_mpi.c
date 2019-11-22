/*!
  \file   tester.c
  \brief  Validate kNN ring implementation.

  \author Dimitris Floros
  \date   2019-11-13
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "knnring.h"

/* #define VERBOSE */

static char * STR_CORRECT_WRONG[] = {"WRONG", "CORRECT"};
struct timespec start, finish;
double elapsed;

// =================
// === UTILITIES ===
// =================

double dist(double *X, double *Y, int i, int j, int d, int n, int m){

  /* compute distance */
  double dist = 0;
  for (int l = 0; l < d; l++){
    dist +=( ( X[i*d+l] - Y[j*d+l] ) * ( X[i*d+l] - Y[j*d+l]  ) );
  }

  return sqrt(dist);
}


// ==================
// === VALIDATION ===
// ==================

//! kNN validator
/*!
   The function asserts correctness of the kNN results by:
     (i)   Checking that reported distances are correct
     (ii)  Validating that distances are sorted in non-decreasing order
     (iii) Ensuring there are no other points closer than the kth neighbor
*/
int validateResult( knnresult knnres, double * corpus, double * query,
                    int n, int m, int d, int k ) {

  /* loop through all query points */
  for (int j = 0; j < m; j++ ){

    /* max distance so far (equal to kth neighbor after nested loop) */
    double maxDist = -1;

    /* mark all distances as not computed */
    int * visited = (int *) calloc( n, sizeof(int) );

    /* go over reported k nearest neighbors */
    for (int i = 0; i < k; i++ ){

      /* keep list of visited neighbors */
      visited[ knnres.nidx[i*m + j] ] = 1;


      /* get distance to stored index */
      double distxy = dist( corpus, query, knnres.nidx[i*m + j], j, d, n, m );

      /* make sure reported distance is correct */
      if ( abs( knnres.ndist[i*m + j] - distxy ) > 1e-8 ) {printf("I BREAK IN 1\n" );
      printf(" distxy : %lf \n" , distxy);
      printf(" knnres.ndist[i*m + j] : %lf \n" , knnres.ndist[i*m + j]);
      printf(" knnres.nidx[i*m + j] : %d \n" , knnres.nidx[i*m + j]);
      printf(" j: %d \n" , j);
      printf(" i: %d \n" , i);



       return 0;}

      /* distances should be non-decreasing */
      if ( knnres.ndist[i*m + j] < maxDist ) {printf("I BREAK IN 2\n" ); return 0;}

      /* update max neighbor distance */
      maxDist = knnres.ndist[i*m + j];

    } /* for (k) -- reported nearest neighbors */
    /* now maxDist should have distance to kth neighbor */

    /* check all un-visited points */
    for (int i = 0; i < n; i++ ){

      /* check only (n-k) non-visited nodes */
      if (!visited[i]){

        /* get distance to unvisited vertex */
        double distxy = dist( corpus, query, i, j, d, n, m );

        /* point cannot be closer than kth distance */
        if ( distxy < maxDist ) {printf("I BREAK IN 3\n" ); return 0;}

      } /* if (!visited[i]) */

    } /* for (i) -- unvisited notes */

    /* deallocate memory */
    free( visited );

  } /* for (j) -- query points */

  /* return */
  return 1;

}


/////VALIDATOR END/////////



////My main
int main(){
  clock_t t;
  double time_taken;

  int n=1000;                    // corpus
  int m=n;                    // query
  int d=300;                      // dimensions
  int k=50;                     // # neighbors
  int i=0;
  int j=0;

  double  * corpus = (double * ) malloc( n*d * sizeof(double) );
  double  * query  = corpus;

  // double(*X)[d] = (double(*)[d])corpus;
  // double(*Y)[d] = (double(*)[d])query;

  for (int i=0;i<n*d;i++)
    corpus[i]= (double)rand()/(double)RAND_MAX;
  for (int i=0;i<m*d;i++)
    query[i]= (double)rand()/(double)RAND_MAX;


  // printf("corpus: \n");
  // for (int i=0;i<n*d;i++)
  //   printf("%f ",corpus[i] );
  // printf("\nquery: \n");
  // for (int i=0;i<m*d;i++)
  //   printf("%f ",query[i] );
  //
  //   printf("Given X:\n");
  //   for ( int i = 0; i < n; i++) {
  //     printf("[ ");
  //     for (int  j = 0; j < d; j++) {
  //       printf("%lf ",corpus[i*d+j]);
  //     }
  //     printf("]\n");
  //   }
  //
  //   printf("Given Y:\n");
  //   for ( i = 0; i < m; i++) {
  //     printf("[ ");
  //     for ( j = 0; j < d; j++) {
  //       printf("%lf ",query[i*d+j]);
  //     }
  //     printf("]\n");
  //   }
  clock_gettime(CLOCK_MONOTONIC, &start);
  // knnresult knnres1 = distrAllkNN( corpus,  n, d, k );
  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed = (finish.tv_sec - start.tv_sec);
  elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  double elapsed2;

  clock_gettime(CLOCK_MONOTONIC, &start);
  knnresult knnres2 = kNN( corpus, corpus, n, n, d, k );
  clock_gettime(CLOCK_MONOTONIC, &finish);
  elapsed2 = (finish.tv_sec - start.tv_sec);
  elapsed2 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

  printf("With Ring :For n=%d , m =%d , d=%d ,k =%d  the time is: %lf seconds! \n",n,m,d,k, elapsed );
  printf("Serial :For n=%d , m =%d , d=%d ,k =%d  the time is: %lf seconds! \n",n,m,d,k, elapsed2 );

  // int isValid1 = validateResult( knnres1, corpus, query, n, m, d, k );
  int isValid2 = validateResult( knnres2, corpus, query, n, m, d, k );

  // printf("Tester validation: %s NEIGHBORS\n", STR_CORRECT_WRONG[isValid1]);
  printf("Tester validation: %s NEIGHBORS\n", STR_CORRECT_WRONG[isValid2]);

  free( corpus );
  free( query );

  return 0;

}
