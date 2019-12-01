#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "knnring.h"
#include "cblas-openblas.h"
#include "limits.h"



  //Getting variables to use as specific blas routines
  enum CBLAS_ORDER order;
  enum CBLAS_TRANSPOSE  transx,transy;

  //Functions Declaration
void ComputeDistancesMatrix(double * D,double * X, double * Y, int n, int m, int d);
void swap_double(double * xp, double * yp);
void swap_int(int * xp, int * yp);
int partition (double * arr,int * idxArr, int low, int high,int ld);
void sortDistancesAndIdxColumns(double * arr,int * idxArr ,int low, int high,int k,int ld);

knnresult kNN(double * X, double * Y, int n, int m, int d, int k){

  //Declaring and allocating the matrixes we will need
  double * D= (double * ) malloc( n*m * sizeof(double) );
  int * idx= (int * ) malloc( n*m * sizeof(int) );

  knnresult result;

  result.ndist =  (double * )malloc(k*m * sizeof(double));
  result.nidx= (int * )malloc(k*m * sizeof(int));
  if( (!D) || (!idx) || (!result.ndist) || (!result.nidx) ){
    printf("Couldn't Allocate Memory!\n" );
    exit(1);
  }

  result.m =m;
  result.k =k;

  //Initialiazing idx values
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      idx[i*m+j]=i;
    }
  }

  //Compute the distances matrix D
  // (D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');)
  ComputeDistancesMatrix(D,X,Y,n ,m ,d);

  //Sorting every column of D and idx matrixes
  for(int i=0 ; i< m ;i++){
    sortDistancesAndIdxColumns((D+i),(idx+i),0,n-1,k, m );
  }

  //reallocating D and idx , in smaller sizes , to avoid excessive use of memory
  D= (double *) realloc(D,k*m * sizeof(double));
  idx=(int *) realloc(idx,k*m * sizeof(int));


  //Getting the m*K Matrix
  for(int i=0; i < k ; i++){
    for(int j =0 ; j<m ;j++)
      result.ndist[j*k+i]=D[i*m+j];
  }
  for(int i=0; i < k ; i++){
    for(int j =0 ; j<m ;j++)
      result.nidx[j*k+i]=idx[i*m+j];
  }

  free(D);
  free(idx);
  return result;
}


//Function to calculate distances
void ComputeDistancesMatrix(double *D,double * X, double * Y, int n, int m, int d){
  // D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');

    // -2 * X * Y.'
    //blas parameters
    double alpha=-2;
    double beta=0;
    int ldx=d;
    int ldy=d;
    int ldEd=m;
    order=CblasRowMajor;
    transx=CblasNoTrans;
    transy=CblasTrans;
    cblas_dgemm(order,transx,transy, n, m, d, alpha, X, ldx, Y, ldy, beta, D, ldEd );

    //Xnorms2 is the variable to store the sum(X.^2,2) elements
    //Ynorms2 is the variable to store the sum(Y.^2,2)) elements
    double xNorms2,yNorms2;

    //sum(X.^2,2) - 2 * X*Y.'
    //sum(X.^2,2
    for(int i=0;i<n;i++){
      xNorms2=0;
      for(int q=0;q<d;q++){
        xNorms2+=X[i*d+q]*X[i*d+q];
      }
      for(int j=0;j<m;j++){
        D[i*m+j]+=xNorms2;
      }
    }

    //sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).')
    //sum(Y.^2,2).'
    for(int j=0;j<m;j++){
      yNorms2=0;
      for(int q=0;q<d;q++){
        yNorms2+=Y[j*d+q]*Y[j*d+q];
      }
      for(int i=0;i<n;i++){
        D[i*m+j]+=yNorms2;
      }
    }

    //D=sqrt(D)
    for(int i=0; i<n*m ; i++){
        D[i]=sqrt(D[i]);
    ////////////If computing the distance between the same spot///////////
    //If we are counting distances between the same point, to avoid a spot be its neighbor
    // if(D[i]<=0 ||isnan(D[i]) ){D[i] =INFINITY;}
    //The code to pass the validator
    if(D[i]<=0 ||isnan(D[i]) ){D[i] =0;}
    }
}
//// WE use quickselect in order to store the properties for the knn
//Swap for using in the quickselect algorithm
void swap_double(double * xp, double * yp)
{
    double  temp = *xp;
    *xp = *yp;
    *yp = temp;
}

void swap_int(int * xp, int * yp)
{
    int  temp = *xp;
    *xp = *yp;
    *yp = temp;
}


///////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
int partition (double * arr,int * idxArr, int low, int high,int ld)
{
    double pivot = arr[high*ld];    // pivot
    int i = (low - 1);  // Index of smaller element

    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j*ld] <= pivot)
        {
            i++;    // increment index of smaller element
            swap_double(&arr[i*ld], &arr[j*ld]);
            swap_int(&idxArr[i*ld], &idxArr[j*ld]);
        }
    }
    swap_double(&arr[(i + 1)*ld], &arr[high*ld]);
    swap_int(&idxArr[(i + 1)*ld], &idxArr[high*ld]);
    return (i + 1);
}

/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
void sortDistancesAndIdxColumns(double * arr,int * idxArr ,int low, int high,int k,int ld)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr,idxArr, low, high,ld);

        // Separately sort elements before
        // partition and after partition
        sortDistancesAndIdxColumns(arr,idxArr, low, pi - 1,k,ld);
        if((pi+1) < k)
          sortDistancesAndIdxColumns(arr,idxArr, pi + 1, high,k,ld);
    }
}
