
#include "imf.h"
#include "smat.h"
#include "dmat.h"
#include "dbilinear.h"

static double norm(double *W, size_t size) {
	double ret = 0;
	for(size_t i = 0; i < size; i++)
		ret += W[i]*W[i];
	return sqrt(ret);
}

void glr_mf_train(glr_mf_prob_t *prob, glr_mf_param_t *param, double *W, double *H, smat_t *testY, double *rmse) { // {{{
	size_t m = prob->m, n = prob->n;
	size_t k = param->k;


	smat_t &Y = *(prob->Y);
	smat_t Yt = Y.transpose();

	glr_prob_t subprob_w(&Yt, H, k, prob->A);
	glr_prob_t subprob_h(&Y, W, k, prob->B);
	glr_solver W_solver(&subprob_w, param);
	glr_solver H_solver(&subprob_h, param);

	if(param->verbose != 0) {
		//printf("|W0| (%ld %ld)= %.6f\n", k, m, norm(W,m*k));
		//printf("|H0| (%ld %ld)= %.6f\n", k, n, norm(H,n*k));
	}

	double Wtime=0, Htime=0, start_time=0;
	for(int iter = 1; iter <= param->maxiter; iter++) {
		omp_set_num_threads(param->threads);
		start_time = omp_get_wtime();
		W_solver.init_prob();
		W_solver.solve(W);
		Wtime += omp_get_wtime()-start_time;

		start_time = omp_get_wtime();
		H_solver.init_prob();
		H_solver.solve(H);
		Htime += omp_get_wtime()-start_time;
		omp_set_num_threads(omp_get_num_procs());
		if(param->verbose != 0) {
			printf("GRMF-iter %d W %.5g H %.5g walltime %.5g", 
					iter, Wtime, Htime, Wtime+Htime);
			double reg_w = trace_dmat_smat_dmat(W,*(prob->A),W,k);
			double reg_h = trace_dmat_smat_dmat(H,*(prob->B),H,k);
			H_solver.init_prob();
			double loss = 2*H_solver.fun(H)-reg_h;
			double obj = 0.5*(loss+reg_w+reg_h);
			printf(" loss %g reg %g obj %g", loss, reg_w+reg_h, obj);
			if(testY) {
				double tmp_rmse = cal_rmse(*testY,W,H,k);
				if(rmse!=NULL) *rmse = tmp_rmse;
				printf(" rmse %lf", tmp_rmse);
			}
			puts("");
			fflush(stdout);
		}
	}
}// }}}

void glr_half_mf_train(glr_mf_prob_t *prob, glr_mf_param_t *param, double lambda, double *W, double *H, smat_t *testY, double *rmse) { // {{{
	size_t m = prob->m, n = prob->n;
	size_t k = param->k;


	smat_t &Y = *(prob->Y);
	smat_t Yt = Y.transpose();

	glr_prob_t subprob_w(&Yt, H, k, prob->A);
	glr_prob_t subprob_h(&Y, W, k, prob->B);
	glr_solver W_solver(&subprob_w, param);
	glr_solver H_solver(&subprob_h, param, lambda);

	if(param->verbose != 0) {
		//printf("|W0| (%ld %ld)= %.6f\n", k, m, norm(W,m*k));
		//printf("|H0| (%ld %ld)= %.6f\n", k, n, norm(H,n*k));
	}

	double Wtime=0, Htime=0, start_time=0;
	for(int iter = 1; iter <= param->maxiter; iter++) {
		omp_set_num_threads(param->threads);
		start_time = omp_get_wtime();
		W_solver.init_prob();
		W_solver.solve(W);
		Wtime += omp_get_wtime()-start_time;

		start_time = omp_get_wtime();
		H_solver.init_prob();
		H_solver.solve(H);
		Htime += omp_get_wtime()-start_time;
		omp_set_num_threads(omp_get_num_procs());
		if(param->verbose != 0) {
			printf("GRMF-iter %d W %.5g H %.5g walltime %.5g", 
					iter, Wtime, Htime, Wtime+Htime);
			double reg_w = trace_dmat_smat_dmat(W,*(prob->A),W,k);
			double reg_h = trace_dmat_smat_dmat(H,*(prob->B),H,k);
			H_solver.init_prob();
			double loss = 2*H_solver.fun(H)-reg_h;
			double obj = 0.5*(loss+reg_w+reg_h);
			printf(" loss %g reg %g obj %g", loss, reg_w+reg_h, obj);
			if(testY) {
				double tmp_rmse = cal_rmse(*testY,W,H,k);
				if(rmse!=NULL) *rmse = tmp_rmse;
				printf(" rmse %lf", tmp_rmse);
			}
			puts("");
			fflush(stdout);
		}
	}
}// }}}
