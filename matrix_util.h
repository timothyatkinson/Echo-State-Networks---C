#ifndef MU_H
#define MU_H

#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include "moore_penrose.h"

/**print_matrix - PRINT MATRIX
  *Prints a gsl_matrix.
    * m. The matrix to print.
*/
void print_matrix(const gsl_matrix *m);

/**print_vector - PRINT VECTOR
  *Prints a gsl_vector.
    * v. The vector to print.
*/
void print_vector (const gsl_vector * v);

/**gsl_matrix_multiply - GSL MATRIX MULTIPLY
  *Multiplies 2 gsl_matrices together a.b, returning the produced matrix as a pointer. Preserves existing matrices. Multiplication is done using the CBLAS function gsl_blas_dgemm.
    * a. The first (left) matrix.
    * b. The second (right) matrix.
*/
gsl_matrix* gsl_matrix_multiply(gsl_matrix* a, gsl_matrix* b);

/**gsl_matrix_pinv - GSL MATRIX PSEUDOINVERSE
  *A non-destructive wrapper for moore_penrose_pinv (see moore_penrose.c).
    * a. The matrix to inverse.
    * rcond. 	A real number specifying the singular value threshold for inclusion. NumPy default for ``rcond`` is 1E-15.
*/
gsl_matrix* gsl_matrix_pinv(gsl_matrix* a, double rcond);

/** gsl_matrix_eigenvalues - GSL MATRIX EIGENVALUES
	* A non-destructive computation of a matrix's eigenvalues using gsl_eigen_symm.
		* The matrix to get the eigenvalues of.
*/
gsl_vector* gsl_matrix_eigen_values(gsl_matrix* a);

/** gsl_matrix_max_eigenvalue - GSL MATRIX MAX EIGENVALUE
	* Computes the maximum eigenvalue of a matrix.
		* a. The matrix to get the maximum eigenvalue of.
*/
double gsl_matrix_max_eigenvalue(gsl_matrix* a);

#endif
