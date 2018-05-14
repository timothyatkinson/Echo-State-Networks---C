#ifndef MP_H
#define MP_H

#include <stdio.h>
#include <gsl/gsl_blas.h>

#include <gsl/gsl_linalg.h>

gsl_matrix* moore_penrose_pinv(gsl_matrix *A, const double rcond);

#endif
