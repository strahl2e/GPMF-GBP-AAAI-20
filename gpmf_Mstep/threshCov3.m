% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

% This code is is based on code from the following reference with modification to handle only checking specific elements on the matrix:
% Author:    Richard Y. Zhang (ryz@alum.mit.edu)
% Url:       http://alum.mit.edu/www/ryz
% Date:      Feb 2018
% Reference: R.Y. Zhang, S. Fattahi, S. Sojoudi, "Linear-Time Algorithm for 
%            Learning Large-Scale Sparse Graphical Models".

%% Positive thresholding of symmetric matrix, omitting the diagonal that is left as is.
% For speed use sparse matrix representation of S.

% Example:
% x=rand(5,3); S=cov(x,1); S_tu=triu(S,1); tau=max(max(S_tu))-0.5*min(min(S_tu(S_tu>0))); pos_only=1;
% S
% tau
% threshCov3(S,tau,pos_only)

function S_thresh = threshCov3(S, tau)
    S_ut = triu(S,1);
    n = size(S,1);
    [ii_Sut,jj_Sut,vv_Sut] = find(S_ut);
    vv_Sut_thresh=max(vv_Sut-tau,0);
    S_thresh_ut = sparse(ii_Sut,jj_Sut,vv_Sut_thresh,n,n);
    S_thresh = S_thresh_ut + S_thresh_ut' + spdiags(diag(S),0,n,n);
end
