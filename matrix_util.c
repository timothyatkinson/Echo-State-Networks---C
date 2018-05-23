#include "matrix_util.h"

/**print_matrix - PRINT MATRIX
  *Prints a gsl_matrix.
    * m. The matrix to print.
*/
void print_matrix(const gsl_matrix *m) {
	size_t i, j;

	for (i = 0; i < m->size1; i++) {
		for (j = 0; j < m->size2; j++) {
			printf("%f\t", gsl_matrix_get(m, i, j));
		}
		printf("\n");
	}
}

/**print_vector - PRINT VECTOR
  *Prints a gsl_vector.
    * v. The vector to print.
*/
void print_vector (const gsl_vector * v) {
	size_t i;

	for (i = 0; i < v->size; i++) {
		printf("%f\t", gsl_vector_get (v, i));
	}
}


/**Multiplies 2 gsl_matrices together a.b, returning the produced matrix as a pointer. Preserves existing matrices. Multiplication is done using the CBLAS function gsl_blas_dgemm.
    * a. The first (left) matrix.
    * b. The second (right) matrix.
*/
gsl_matrix* gsl_matrix_multiply(gsl_matrix* a, gsl_matrix* b){

	int a1 = a->size1;
	int a2 = a->size2;
	int b1 = b->size1;
	int b2 = b->size2;

	if(a2 != b1){
		printf("Error: Matrix dimensions do not match. %d x %d . %d x %d\n", a1, a2, b1, b2);
		//exit(0);
	}

	gsl_matrix* c = gsl_matrix_alloc(a1, b2);

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, a, b, 0.0, c);

	return c;
}

/**Multiplies 2 gsl_matrices together at.b, returning the produced matrix as a pointer. Preserves existing matrices. Multiplication is done using the CBLAS function gsl_blas_dgemm.
    * a. The first (left) matrix. This is transposed.
    * b. The second (right) matrix.
*/
gsl_matrix* gsl_matrix_multiply_transpose_a(gsl_matrix* a, gsl_matrix* b){

	int a1 = a->size1;
	int a2 = a->size2;
	int b1 = b->size1;
	int b2 = b->size2;

	if(a1 != b1){
		printf("Error: Matrix dimensions do not match. %d x %d . %d x %d\n", a1, a2, b1, b2);
		//exit(0);
	}

	gsl_matrix* c = gsl_matrix_alloc(a2, b2);

	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, a, b, 0.0, c);

	return c;
}

/**Multiplies 2 gsl_matrices together a.bt, returning the produced matrix as a pointer. Preserves existing matrices. Multiplication is done using the CBLAS function gsl_blas_dgemm.
    * a. The first (left) matrix.
    * b. The second (right) matrix. This is transposed.
*/
gsl_matrix* gsl_matrix_multiply_transpose_b(gsl_matrix* a, gsl_matrix* b){

	int a1 = a->size1;
	int a2 = a->size2;
	int b1 = b->size1;
	int b2 = b->size2;

	if(a2 != b2){
		printf("Error: Matrix dimensions do not match. %d x %d . %d x %d\n", a1, a2, b1, b2);
		//exit(0);
	}

	gsl_matrix* c = gsl_matrix_alloc(a1, b1);

	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, a, b, 0.0, c);

	return c;
}

/**gsl_matrix_det - GSL MATRIX DET
	*Computes the determinant of a matrix using its LU decomposition.
		*a. The matrix to get the determinant of.
*/
double gsl_matrix_det(gsl_matrix* a){
	int signum;

	gsl_matrix* LU = gsl_matrix_alloc(a->size1, a->size2);
	gsl_matrix_memcpy(LU, a);

	gsl_permutation* p = gsl_permutation_alloc(a->size1);

	gsl_linalg_LU_decomp(LU, p, &signum);

	double det = gsl_linalg_LU_det(LU, signum);

	gsl_matrix_free(LU);
	gsl_permutation_free(p);

	return det;
}

/**gsl_matrix_inverse - GSL MATRIX INVERSE
	*Computes the inverse of a matrix based on its LU decomposition.
		*a. The matrix to inverse.
*/
gsl_matrix* gsl_matrix_inverse(gsl_matrix* a){
	int signum;

	gsl_matrix* LU = gsl_matrix_alloc(a->size1, a->size2);
	gsl_matrix_memcpy(LU, a);

	gsl_permutation* p = gsl_permutation_alloc(a->size1);

	gsl_linalg_LU_decomp(LU, p, &signum);

	gsl_matrix* inverse = gsl_matrix_alloc(a->size2, a->size1);

	gsl_linalg_LU_invert(LU, p, inverse);

	gsl_matrix_free(LU);
	gsl_permutation_free(p);

	return inverse;
}

/**gsl_matrix_pinv - GSL MATRIX PSEUDOINVERSE
  *A non-destructive wrapper for moore_penrose_pinv (see moore_penrose.c). moore_penrose_pinv destroys its input argument - this wrapper copies the input argument and frees the
	*copy after the pseudoinverse takes place, preserving the original input argument.
    * a. The matrix to inverse.
    * rcond. 	A real number specifying the singular value threshold for inclusion. NumPy default for ``rcond`` is 1E-15.
*/
gsl_matrix* gsl_matrix_pinv(gsl_matrix* a, double rcond){
	gsl_matrix* new = gsl_matrix_alloc(a->size1, a->size2);
	gsl_matrix_memcpy(new, a);
	gsl_matrix* pinv = moore_penrose_pinv(new, rcond);
	gsl_matrix_free(new);
	return pinv;
}

/** gsl_matrix_eigenvalues - GSL MATRIX EIGENVALUES
	* A non-destructive computation of a matrix's eigenvalues using gsl_eigen_symm.
		* a. The matrix to get the eigenvalues of.
*/
gsl_vector* gsl_matrix_eigen_values(gsl_matrix* a){
	gsl_vector* k = gsl_vector_alloc(a->size1);
	gsl_eigen_symm_workspace* w = gsl_eigen_symm_alloc(a->size1);
	gsl_eigen_symm(a, k, w);
	gsl_eigen_symm_free(w);
	return k;
}

/** gsl_matrix_max_eigenvalue - GSL MATRIX MAX EIGENVALUE
	* Computes the maximum eigenvalue of a matrix.
		* a. The matrix to get the maximum eigenvalue of.
*/
double gsl_matrix_max_eigenvalue(gsl_matrix* a){
	gsl_vector* eigen = gsl_matrix_eigen_values(a);
	double max = abs(gsl_vector_get(eigen, 0));
	for(int i = 1; i < a->size1; i++){
		if(abs(gsl_vector_get(eigen, i)) > max){
			max = abs(gsl_vector_get(eigen, i));
		}
	}
	gsl_vector_free(eigen);
	return max;
}
