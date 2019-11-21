#ifndef IMF_H
#define IMF_H

#include "dbilinear.h"

struct imf_prob_t { // {{{
	smat_t *Y;
	void *A;     // row features
	void *B;     // column features
	size_t m, n;   // dimension of Y
	size_t fa, fb; // #features of A and B
	size_t k;      // #topics 
	int A_type, B_type; // type of A and B: 0 for dense, 1 for sparse, 2 for Identity
	enum {Dense=0, Sparse=1, Identity=2}; // same as bilinear_prob_t
	imf_prob_t(smat_t *Y, void *A, void *B, size_t fa, size_t fb, size_t k, int A_type=0, int B_type=0): 
		Y(Y), A(A), B(B), m(Y->rows), n(Y->cols), fa(fa), fb(fb), k(k), A_type(A_type), B_type(B_type) {}
};
struct imf_param_t : public bilinear_param_t {
	size_t k;
	int maxiter;
	int top_p;
	int predict;
	imf_param_t(): bilinear_param_t() {
		k = 10;
		maxiter = 10;
		top_p = 10;
		predict = 0;
	}
}; // }}}

struct glr_mf_prob_t { // {{{
	smat_t *Y;
	smat_t *A;     // Laplacian for rows
	smat_t *B;     // Laplacian for cols 
	size_t m, n;   // dimension of Y
	size_t k;      // #topics 
	glr_mf_prob_t(smat_t *Y, smat_t *A, smat_t *B, size_t k): 
		Y(Y), A(A), B(B), m(Y->rows), n(Y->cols), k(k){}
};
struct glr_mf_param_t : public glr_param_t {
	size_t k;
	int maxiter;
	int top_p;
	int predict;
	glr_mf_param_t(): glr_param_t() {
		k = 10;
		maxiter = 10;
		top_p = 10;
		predict = 0;
	}
}; // }}}
#ifdef __cplusplus
extern "C" {
#endif

void glr_mf_train(glr_mf_prob_t *prob, glr_mf_param_t *param, double *W, double *H, smat_t *testY=NULL, double *rmse=NULL);
void glr_half_mf_train(glr_mf_prob_t *prob, glr_mf_param_t *param, double lambda, double *W, double *H, smat_t *testY=NULL, double *rmse=NULL);

#ifdef __cplusplus
}
#endif

#endif /*IMF_H*/

/* 
class mf_parameter: public bilinear_parameter{ 
	public:
		int maxiter;
		int top_p;
		int k;
		int threads;
		int reweighting;
		bool predict;
		// Parameters for Wsabie
		double lrate; // learning rate for wsabie
		mf_parameter() {
			bilinear_parameter();
			reweighting = 0;
			maxiter = 10; 
			top_p = 20; 
			k = 10;
			threads = 8;
			lrate = 0.01;
			predict = true;
		}
};
*/

