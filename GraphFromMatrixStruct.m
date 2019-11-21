% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

function [G] = GraphFromMatrixStruct(M)
    % Given an arbitrary sparse symmetric matrix, M, return an undirected graph
    % from the sparsity structure.
    dimM = size(M,1);
    [i,j] = find(triu(M,1));
    S = sparse(i,j,ones(length(i),1),dimM,dimM);
    G = S + S';
end
