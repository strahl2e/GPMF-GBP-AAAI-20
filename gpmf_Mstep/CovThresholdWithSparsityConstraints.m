% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.


function [C_thresh_const] = CovThresholdWithSparsityConstraints(M_step_cov,gamma,X)
% Positive thresholding of covaraince matrix for sparsity structure.
% Drop contested edges (when gamma=0)

M_step_cov_minusDiag = M_step_cov - spdiags(diag(M_step_cov),0,size(M_step_cov,1),size(M_step_cov,1));
S_sparse_max = max(max(M_step_cov_minusDiag)); %Find max positive off-diagonal value.

%https://arxiv.org/pdf/1504.02995.pdf  Thresholding should dominate the
%maximum estimation error ... C * sqrt(log p / n) with large enough C.
tau= gamma*S_sparse_max* sqrt(log(size(X,2)) / size(X,1));
C_thresh_const = threshCov3(M_step_cov, tau);

end

