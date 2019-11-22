#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "knnring.h"
#include "cblas-openblas.h"
#include "limits.h"





// knnresult kNN(double * X, double * Y, int n, int m, int d, int k);
//! Compute distributed all-kNN of points in X
/*!
\param X Data points [n-by-d]
\param n Number of data points [scalar]
\param d Number of dimensions [scalar]
\param k Number of neighbors [scalar]
\return The kNN result
*/
knnresult distrAllkNN(double * X, int n, int d, int k){




}
