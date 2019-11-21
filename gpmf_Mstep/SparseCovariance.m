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

%% Compute sample covariance elements limited to non-zero locations in graph G
% Result is sample covariance matrix S with same zero pattern as G.

% e.g.
% X = randn(10,5); G_ut=sparse([1,1,1,2,2,3],[3,4,5,4,5,5],ones(6,1),5,5);
% G = G_ut + G_ut';
% cov(X-mean(X),1)
% full(SparseCovariance(X-mean(X),G))
% full(G)

function [S_sparse] = SparseCovariance(X,G)
    [n,p] = size(X);
    [Gi_triu,Gj_triu] = find(triu(G,1) + spdiags(ones(p,1),0,p,p));
    Xis = X(:,Gi_triu);
    Xjs = X(:,Gj_triu);
    S_tri = sparse(Gi_triu, Gj_triu, sum(Xis.*Xjs,1)./n);
    S_sparse = S_tri + triu(S_tri,1)';
end
