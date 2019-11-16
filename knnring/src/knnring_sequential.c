#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "knnring.h"
#include "cblas.h"
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
  //Declaring the matrix of the distances D
  double * D= (double * ) malloc( n*m * sizeof(double) );
  if(!D){
    printf("Couldn't Allocate Memory!\n" );
    exit(1);
  }

  //Declaring the the idx matrix
  int * idx= (int * ) malloc( n*m * sizeof(int) );
  if(!idx){
    printf("Couldn't Allocate Memory!\n" );
    exit(1);
  }
  //initialiazing idx values
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      idx[i*m+j]=i;
    }
  }
  //test idx
  //test Ynorms
  printf("IDX:\n" );
  for (int i = 0; i < n*m; i++) {
    printf("[ ");
    // for ( j = 0; j < m; j++) {
      printf("%d ",idx[i]);
    // }
    printf("]\n");
  }

  //Compute the distances matrix D
  // (D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');)
  ComputeDistancesMatrix(D,X,Y,n ,m ,d);

  //double kthSmallest(double arr[], int l, int r, int k)
  //void sortDistancesAndIdxColumns(double * arr,int * idxArr ,int low, int high,int k,int ld);
  for(int i=0 ; i< m ;i++){
    sortDistancesAndIdxColumns((D+i),(idx+i),0,n-1,k, m );
  }


    printf("I did the sorting\n");
    //test new D , idx
    printf("NEW D  before realloc:\n" );
    for (int  i = 0; i < n; i++) {
      printf("[ ");
      for (int  j = 0; j < m; j++) {
        printf("%lf ",D[i*m+j]);
      }
      printf("]\n");
    }



  // D= (double *) realloc(D,k*m * sizeof(double));
  // idx=(int *) realloc(idx,k*m * sizeof(int));



  //declaring result
  knnresult result;
  // if(!result){
  //   printf("Couldn't Allocate Memory!\n" );
  //   exit(1);
  // }

  //test new D , idx
  printf("NEW D after realloc:\n" );
  for (int i = 0; i < k; i++) {
    printf("[ ");
    for (int  j = 0; j < m; j++) {
      printf("%lf ",D[i*m+j]);
    }
    printf("]\n");
  }

  //test new D , idx
  printf("NEW D after realloc:\n" );
  for (int i = 0; i < n; i++) {
    printf("[ ");
    for (int  j = 0; j < m; j++) {
      printf("%lf ",D[i*m+j]);
    }
    printf("]\n");
  }

    printf("NEW idx :\n" );
  for (int  i = 0; i < n; i++) {
    printf("[ ");
    for ( int j = 0; j < m; j++) {
      printf("%d ",idx[i*m+j]);
    }
    printf("]\n");
  }
  result. ndist =  (double *)malloc(k*m * sizeof(double));
  result. nidx= (int *)malloc(k*m * sizeof(int));

  // for(int i=0; i < k ; i++){
  //   for(int j =0 ; j<m ;j++)
  //     result. ndist[j*k+i]=D[i*m+j];
  // }
  //
  // for(int i=0; i < k ; i++){
  //   for(int j =0 ; j<m ;j++)
  //     result. nidx[j*k+i]=idx[i*m+j];
  // }


  for(int i=0; i < k*m ; i++){
      result. ndist[i]=D[i];
  }

  for(int i=0; i < k*m ; i++){
      result. nidx[i]=idx[i];
  }
  // result. nidx = idx;
  // result. ndist= D;
  result . m =m;
  result . k =k;

  //test new D , idx
  printf(" result NEW D after realloc:\n" );
  for (int i = 0; i < k; i++) {
    printf("[ ");
    for (int  j = 0; j < m; j++) {
      printf("%lf ",  result. ndist[i*m+j]);
    }
    printf("]\n");
  }
    printf("REST idx :\n" );
  for (int  i = 0; i < k; i++) {
    printf("[ ");
    for ( int j = 0; j < m; j++) {
      printf("%d ",result.nidx[i*m+j]);
    }
    printf("]\n");
  }





  free(D);
  free(idx);
  return result;
}





void ComputeDistancesMatrix(double *D,double * X, double * Y, int n, int m, int d){
  // D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');

      // sum(X.^2,2)
        //X.^2
      double * xPow2=(double *)malloc(n*d*sizeof(double));
      if(!xPow2){
        printf("Couldn't Allocate Memory!\n" );
        exit(1);
      }

      //getting the x^2
      for(int i=0;i<n*d;i++){
        xPow2[i]=pow( X[i],2);
      }

        // sum(X.^2,2)
      double * xNorms2=(double *)malloc(n*sizeof(double));
      if(!xNorms2){ //checking for memory allocation
        printf("Couldn't Allocate Memory!\n" );
        exit(1);
      }

      // compute the norms (^2) of the points in X
      for(int i=0;i<n;i++){
        xNorms2[i]=0;
        for(int j=0;j<d;j++){
          xNorms2[i]+=xPow2[i*d+j];
        }
      }
      free(xPow2);

      //test Xnorms
      printf("Xnorms:\n" );
      for (int i = 0; i < n; i++) {
        printf("[ ");
        // for ( j = 0; j < m; j++) {
          printf("%lf ",xNorms2[i]);
        // }
        printf("]\n");
      }

      // sum(Y.^2,2)
        //Y.^2
      double * yPow2=(double *)malloc(m*d*sizeof(double));
      if(!yPow2){
        printf("Couldn't Allocate Memory!\n" );
        exit(1);
      }

      for(int i=0;i<m*d;i++){
        yPow2[i]=pow( Y[i],2);
      }

      // sum(Υ.^2,2)
      double * yNorms2=(double *)malloc(m*sizeof(double));
      if(!yNorms2){
        printf("Couldn't Allocate Memory!\n" );
        exit(1);
      }
      // compute the norms (^2) of the points in Y
      for(int i=0;i<m;i++){
        yNorms2[i]=0;
        for(int j=0;j<d;j++){
          yNorms2[i]+=yPow2[i*d+j];
        }
      }
      free(yPow2);

      //test Ynorms
      printf("Ynorms:\n" );
      for (int i = 0; i < m; i++) {
        printf("[ ");
        // for ( j = 0; j < m; j++) {
          printf("%lf ",yNorms2[i]);
        // }
        printf("]\n");
      }



      // -2 * X * Y.'
      double  * euclideanDistances = (double * ) malloc( n*m * sizeof(double) );
      if(!euclideanDistances){
        printf("Couldn't Allocate Memory!\n" );
        exit(1);
      }
      //blas parameters
      double alpha=-2;
      double beta=0;
      int ldx=d;
      int ldy=d;
      int ldEd=m;
      order=CblasRowMajor;
      transx=CblasNoTrans;
      transy=CblasTrans;
      cblas_dgemm(order,transx,transy, n, m, d, alpha, X, ldx, Y, ldy, beta, euclideanDistances, ldEd );


    //test
    printf("euclideanDistances arr:\n" );
    for (int i = 0; i < n*m; i++) {
      printf("[ ");
      // for ( j = 0; j < m; j++) {
        printf("%lf ",euclideanDistances[i]);
      // }
      printf("]\n");
    }

    //D = sqrt(sum(X.^2,2) - 2 * X*Y.' + sum(Y.^2,2).');
    //Computind each cell in D array
    for(int i=0; i<n ; i++){
      for (int j = 0; j < m; j++) {
        D[i*m+j]=sqrt(xNorms2[i]+ euclideanDistances[i*m+j] + yNorms2[j]);
      }
    }
    //free some memory from the heap
    free(euclideanDistances);
    free(xNorms2);
    free(yNorms2);

    //test D
    printf("D:\n" );
    for (int i = 0; i < n*m; i++) {
      printf("[ ");
      // for ( j = 0; j < m; j++) {
        printf("%lf ",D[i]);
      // }
      printf("]\n");
    }

    printf("D 2d:\n" );
    for ( int i = 0; i < n; i++) {
      printf("[ ");
      for (int  j = 0; j < m; j++) {
        printf("%lf ",D[i*m+j]);
      }
      printf("]\n");
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

/*
// int partition(double * arr,int * idxArr ,int l, int r, int ld)
// {
//
//     double x = arr[r*ld];
//     int i = l;
//     for (int j = l; j <= (r - 1); j++) {
//         if (arr[j*ld] < x) {
//             swap_double(&arr[i*ld],&arr[j*ld]);
//             swap_int(&idxArr[i*ld],&idxArr[j*ld]);
//             i++;
//         }
//     }
//     swap_double(&arr[i*ld],&arr[r*ld]);
//     swap_int(&idxArr[i*ld],&idxArr[r*ld]);
//     return i;
// }

// void sortDistancesAndIdxColumns(double * arr,int * idx, int l, int r,int k,int ld)
//   {
//
//       // If k is smaller than number of
//       // elements in array
//       if ( (k > 0) && k <= ( r - l + 1) ) {
//
//           // Partition the array around last
//           //element and get position of pivot
//           // element in sorted array
//           int index = partition(arr, idx ,l, r,ld);
//           // if(l==r)
//           //   return;
//
//           // If position is same as k
//           // if ((index - l) == (k - 1))
//           //     return ;
//
//           // If position is more, recur
//           // for left subarray
//           if ((index - l )> (k - 1))
//           sortDistancesAndIdxColumns(arr,idx, l, index - 1, k,ld);
//
//
//           // Else recur for right subarray
//           sortDistancesAndIdxColumns(arr,idx, index + 1, r,k - index + l - 1,ld);
//
//       }
//
//       // If k is more than number of
//       // elements in array
//       return;
// }
*/
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
