
#include <time.h>
#include <cstddef>
#include "tron.h"
#include "smat.h"
#include "dmat.h"
#include "dbilinear.h"

#define INF HUGE_VAL

static void print_string_stdout(const char *s) { fputs(s,stdout); fflush(stdout); }
static void print_null(const char *){}
typedef void (*print_fun_ptr)(const char *);
template<typename T>
static print_fun_ptr get_print_fun(T *param) {
//static void (*(get_print_fun(bilinear_param_t *param)))(const char *) {
	if(param->verbose == 0) return print_null;
	else if(param->verbose == 1) return print_null;
	else return print_string_stdout;
}

// graph laplacian regularizaiton + squared-L2 loss + full observation + dense X
class glr_ls_fY_dX: public function { // {{{
	const glr_prob_t *prob;
	const glr_param_t *param;
	smat_t &Y, &L;
	double *X;
	size_t m, l, k;
	double trYTY;
	double *YTX; // Y^T*X
	double *XTX; // X^TX
	double *WTW;
	double *z;
	double *D;

	public:
	int get_nr_variable(void) {return(int) (prob->l*prob->k);}
	glr_ls_fY_dX(const glr_prob_t *prob, const glr_param_t *param): 
		prob(prob), param(param),
		Y(*prob->Y), L(*prob->L), X(prob->X),
		m(prob->m), l(prob->l), k(prob->k) { 
		YTX = MALLOC(double, l*k);
		XTX = MALLOC(double, k*k);
		WTW = MALLOC(double, k*k);
		trYTY = do_dot_product(Y.val_t, Y.val_t, Y.nnz);
		//init(); // required init() before solve.
	}
	~glr_ls_fY_dX() {
		if(YTX) free(YTX);
		if(XTX) free(XTX);
		if(WTW) free(WTW);
	}
	void init() {
		smat_t Yt = Y.transpose();
		smat_x_dmat(Yt, X, k, YTX);
		doHTH(X,XTX,m,k);
	}
	double fun(double *w) {
		double f = 0;
		f += 0.5*trYTY;
		doHTH(w, WTW, l, k);
		f += 0.5*do_dot_product(WTW, XTX, k*k);
		f -= do_dot_product(YTX, w, l*k);
		f += 0.5*trace_dmat_smat_dmat(w, L, w, k);
		return f;
	}
	void grad(double *w, double *g) {
		do_copy(YTX, g, l*k);
		doVM(1.0, w, XTX, -1.0, g, l, k);
		smat_x_dmat(1.0, L, w, k, g, g);
	}
	void Hv(double *s, double *Hs) {
		doVM(1.0, s, XTX, 0.0, Hs, l, k);
		smat_x_dmat(1.0, L, s, k,  Hs, Hs);
	}
}; // }}}

// Y with Missing Values
// routines for Y with missing values
void barXv_withXV(const smat_t &Y, double *XV, double *H, size_t k, double *barXv){ // {{{
	// barXv(i,j) = <(XV)(i,1:k), H(j,1:k), forall (i,j) in \Omega(Y)
#pragma omp parallel for schedule(dynamic,32)
	for(size_t i = 0; i < Y.rows; i++) {
		for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
			double sum = 0;
			size_t j = Y.col_idx[idx];
			for(size_t s = 0; s < k; s++) 
				sum += XV[i*k+s]*H[j*k+s];
			barXv[idx] = sum;
		}
	}
} // }}}

class glr_ls_mY_dX: public function { // {{{
	const glr_prob_t *prob;
	const glr_param_t *param;
	smat_t &Y, &L;
	double *X;
	size_t m, l, k;

	public:
	int get_nr_variable(void) {return(int) (prob->l*prob->k);}
	glr_ls_mY_dX(const glr_prob_t *prob, const glr_param_t *param): 
		prob(prob), param(param),
		Y(*prob->Y), L(*prob->L), X(prob->X),
		m(prob->m), l(prob->l), k(prob->k) {}
	~glr_ls_mY_dX() {}
	void init() {}
	double fun(double *w) {
		double f = 0;
#pragma omp parallel for schedule(dynamic,32) reduction(+:f)
		for(size_t i = 0; i < Y.rows; i++) {
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				size_t j = Y.col_idx[idx];
				double sum = -Y.val_t[idx];
				for(size_t t = 0; t < k; t++)
					sum += X[i*k+t]*w[j*k+t];
				f += sum*sum;
			}
		}
		f += trace_dmat_smat_dmat(w, L, w, k);
		f *= 0.5;
		return f;
	}
	void grad(double *w, double *g) {
#pragma omp parallel for schedule(dynamic,32)
		for(size_t j = 0; j < Y.cols; j++) {
			for(size_t t = 0; t < k; t++)
				g[j*k+t] = 0;
			for(long idx = Y.col_ptr[j]; idx != Y.col_ptr[j+1]; idx++) {
				size_t i = Y.row_idx[idx];
				double sum = -Y.val[idx];
				for(size_t t = 0; t < k; t++)
					sum += X[i*k+t]*w[j*k+t];
				for(size_t t = 0; t < k; t++)
					g[j*k+t] += sum*X[i*k+t];
			}
		}
		smat_x_dmat(1.0, L, w, k, g, g);
	}
	void Hv(double *s, double *Hs) {
#pragma omp parallel for schedule(dynamic,32)
		for(size_t j = 0; j < Y.cols; j++) {
			for(size_t t = 0; t < k; t++)
				Hs[j*k+t] = 0;
			for(long idx = Y.col_ptr[j]; idx != Y.col_ptr[j+1]; idx++) {
				size_t i = Y.row_idx[idx];
				double sum = 0;
				for(size_t t = 0; t < k; t++)
					sum += X[i*k+t]*s[j*k+t];
				for(size_t t = 0; t < k; t++)
					Hs[j*k+t] += sum*X[i*k+t];
			}
		}
		smat_x_dmat(1.0, L, s, k, Hs, Hs);
	}
}; // }}}
/*
 *  Case with X = I
 *  W = argmin_{W} 0.5*|Y - W*H'|^2 + 0.5*lambda*|W|^2
 *  W = argmin_{W}  C * |Y - W*H'|^2 +  0.5*|W|^2
 *    C = 1/(2*lambda)
 */
struct l2r_ls_fY_IX_chol: public solver_t {// {{{
	smat_t *Y;
	double *H, *HTH, lambda;
	double *YH, *kk_buf, trYTY; // for calculation of fun()
	size_t m, k;
	bool done_init;

	l2r_ls_fY_IX_chol(bilinear_prob_t *prob, bilinear_param_t *param):
		Y(prob->Y), H(prob->H), HTH(NULL), lambda(param->lambda), YH(NULL), kk_buf(NULL), 
		m(prob->m), k(prob->k), done_init(false) { 
		HTH = MALLOC(double, k*k); 
		YH = MALLOC(double, m*k);
		kk_buf = MALLOC(double, k*k);
		trYTY = do_dot_product(Y->val, Y->val, Y->nnz);
	}
	~l2r_ls_fY_IX_chol() { if(HTH) free(HTH); if(YH) free(YH); if(kk_buf) free(kk_buf);}
	void init_prob() {
		doHTH(H, HTH, Y->cols, k);
		for(size_t t= 0; t < k; t++)
			HTH[t*k+t] += lambda;
		smat_x_dmat(*Y, H, k, YH);
		done_init = true;
	}
	void solve(double *W) {
		if(!done_init) {init_prob();}
		do_copy(YH, W, m*k);
		ls_solve_chol_matrix(HTH, W, k, m);
		done_init = false; // ls_solve_chol modifies HTH...
	}
	double fun(double *w) {
		if(!done_init) {init_prob();}
		double obj = 0;
		obj += trYTY;
		doHTH(w, kk_buf, m, k);
		obj += do_dot_product(kk_buf, HTH, k*k);
		obj -= 2.0*do_dot_product(w, YH, m*k);
		return 0.5*obj;
	}
}; // }}}
struct l2r_ls_mY_IX_chol: public solver_t { // {{{
	smat_t *Y;
	double *H, **Hessian_set, lambda;
	size_t m, k, nr_threads;

	l2r_ls_mY_IX_chol(bilinear_prob_t *prob, bilinear_param_t *param):
		Y(prob->Y), H(prob->H), Hessian_set(NULL), lambda(param->lambda),
		m(prob->m), k(prob->k) {//{{{
		nr_threads = omp_get_max_threads();
		Hessian_set = MALLOC(double*, nr_threads);
		for(size_t i = 0; i < nr_threads; i++)
			Hessian_set[i] = MALLOC(double, k*k);
	} // }}}
	~l2r_ls_mY_IX_chol(){ // {{{
		for(size_t i = 0; i < nr_threads; i++)
			if(Hessian_set[i]) free(Hessian_set[i]);
		free(Hessian_set);
	} // }}}
	void init_prob() {}
	void solve(double *W) { // {{{
		smat_t &Y = *(this->Y);
#pragma omp parallel for schedule(dynamic,64)
		for(size_t i = 0; i < Y.rows; ++i) {
			size_t nnz_i = Y.nnz_of_row(i); 
			if(!nnz_i) continue;
			int tid = omp_get_thread_num(); // thread ID
			double *Wi = &W[i*k];
			double *Hessian = Hessian_set[tid];
			double *y = Wi; 
			memset(Hessian, 0, sizeof(double)*k*k);
			memset(y, 0, sizeof(double)*k);

			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; ++idx) {
				const double *Hj = &H[k*Y.col_idx[idx]];
				for(size_t s = 0; s < k; ++s) {
					y[s] += Y.val_t[idx]*Hj[s];
					for(size_t t = s; t < k; ++t)
						Hessian[s*k+t] += Hj[s]*Hj[t];
				}
			}
			for(size_t s = 0; s < k; ++s) {
				for(size_t t = 0; t < s; ++t)
					Hessian[s*k+t] = Hessian[t*k+s];
				Hessian[s*k+s] += lambda;
			}
			ls_solve_chol_matrix(Hessian, y, k);
		}
	} // }}}
	double fun(double *W) {
		smat_t &Y = *(this->Y);
		double loss = 0;
#pragma omp parallel for reduction(+:loss) schedule(dynamic,32)
		for(size_t i = 0; i < Y.rows; i++) {
			for(long idx = Y.row_ptr[i]; idx != Y.row_ptr[i+1]; idx++) {
				double err = -Y.val_t[idx];
				size_t j = Y.col_idx[idx];
				for(size_t s = 0; s < k; s++)
					err += W[i*k+s]*H[j*k+s];
				loss += err*err;
			}
		}
		double reg = do_dot_product(W,W,m*k);
		return 0.5*(loss + lambda*reg);
	}
}; // }}}

glr_solver::glr_solver(glr_prob_t *prob, glr_param_t *param, double lambda): prob(prob), param(param), lambda(lambda), fun_obj(NULL), tron_obj(NULL), solver_obj(NULL), biprob(NULL), biparam(NULL), done_init(false) { // {{{
	if(lambda <= 0) {
		switch(param->solver_type) {
			case GLR_LS_FULL:
				fun_obj = new glr_ls_fY_dX(prob, param);
				break;
			case GLR_LS_MISSING:
				fun_obj = new glr_ls_mY_dX(prob, param);
				break;
			default:
				fprintf(stderr, "Solver not supported\n");
				break;
		}
		fflush(stdout);
		int max_cg_iter = param->max_cg_iter;
		if(max_cg_iter >= fun_obj->get_nr_variable())
			max_cg_iter = fun_obj->get_nr_variable();
		printf("max_cg_iter %d\n", max_cg_iter);
		tron_obj = new TRON(fun_obj, param->eps, param->max_tron_iter, max_cg_iter);
		tron_obj->set_print_string(get_print_fun(param));
	} else { // lambda > 0 => lambda * Identity
		Yt = prob->Y->transpose();
		biprob = new bilinear_prob_t(&Yt, NULL, prob->X, Yt.rows, prob->k, bilinear_prob_t::Identity);
		biparam = new bilinear_param_t();
		biparam->lambda = lambda;
		biparam->use_chol = 1;
		biparam->threads = param->threads;
		biparam->verbose = param->verbose;
		
		switch(param->solver_type) {
			case GLR_LS_FULL:
				biparam->solver_type = L2R_LS_FULL;
				solver_obj = new l2r_ls_fY_IX_chol(biprob, biparam);
				break;
			case GLR_LS_MISSING:
				biparam->solver_type = L2R_LS_MISSING;
				solver_obj = new l2r_ls_mY_IX_chol(biprob, biparam);
				break;
			default :
				fprintf(stderr, "Solver not supported\n");
				break;
		}
	}
} //}}}

void glr_train(const glr_prob_t *prob, const glr_param_t *param, double *w, double *walltime, double *cputime) { // {{{
	double eps = param->eps;
	double time_start = omp_get_wtime();
	clock_t clock_start = clock();
	if(prob->W)
		do_copy(prob->W, w, prob->l*prob->k);
	function *fun_obj=NULL;
	switch(param->solver_type) {
		case GLR_LS_FULL:
			{
				fun_obj = new glr_ls_fY_dX(prob, param);
				TRON tron_obj(fun_obj, eps, param->max_tron_iter, param->max_cg_iter);
				tron_obj.set_print_string(get_print_fun(param));
				tron_obj.tron(w, false); // prob->W is the inital
				delete fun_obj;
				break;
			}
	}
	if(walltime) *walltime += (omp_get_wtime() - time_start);
	if(cputime) *cputime += ((double)(clock()-clock_start)/CLOCKS_PER_SEC);

	/*
	size_t nnz_Y = prob->Y->nnz;
	double primal_solver_regression_tol = eps*(double)std::max(nnz_Y, 1UL)/(double)(prob->Y->cols*prob->Y->rows);
	*/
} // }}}
double cal_rmse(smat_t &testY, double *W, double *H, size_t k) { // {{{
	double rmse = 0.0;
#pragma omp parallel for schedule(dynamic,50) shared(testY) reduction(+:rmse)
	for(size_t i = 0; i < testY.rows; i++) {
		for(long idx = testY.row_ptr[i]; idx != testY.row_ptr[i+1]; idx++) {
			size_t j = testY.col_idx[idx];
			double true_val = testY.val_t[idx], pred_val = 0.0;
			for(size_t t = 0; t < k; t++)
				pred_val += W[k*i+t]*H[k*j+t];
			rmse += (pred_val-true_val)*(pred_val-true_val);
		}
	}
	return sqrt(rmse/(double)testY.nnz);
} // }}}

