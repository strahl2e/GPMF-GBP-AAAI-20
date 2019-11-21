function [ X ] = genSamp( C, m )
%GENSAMP Generate samples according to a sparse inverse covariance matrix

% Input: C      -- n x n sparse inverse covariance matrix
%        m      -- number of samples
% Ouput: X      -- n x m matrix of samples, each column is sampled i.i.d
%                  from N(0, inv(C))

% Author:    Richard Y. Zhang (ryz@alum.mit.edu)
% Url:       http://alum.mit.edu/www/ryz
% Date:      Feb 2018
% Reference: R.Y. Zhang, S. Fattahi, S. Sojoudi, "Linear-Time Algorithm for 
%            Learning Large-Scale Sparse Graphical Models".

% Updates:   Jonathan Strahl, 
%            Nov 2019, introduce sampling from constrained matrix, at fixed subset of elements i,j
%            Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.


gensampT=tic;
disp('genSamples function....')
toc(gensampT)

n = size(C,1);
assert(size(C,2) == n, 'Inverse covariance must be square');
assert(norm(C - C' ,'fro') == 0, 'Inverse covariance must be perfectly symmetric');
assert(isreal(C), 'Inverse covariance must be real');

% In case some rows/cols have no data we exclude them from the
% covarince estimation then pad out the result before returning.
zero_nodes=find(diag(C) == 0);
if ~isempty(zero_nodes)
    zero_node_idxs=sub2ind([n,n],zero_nodes,zero_nodes);
    diag_C=diag(C);
    diag_C=diag_C(setdiff(1:n,zero_nodes));
    smallest_prec = min(diag_C);
    smallest_prec
    C(zero_node_idxs)=0.1*smallest_prec;
end

% Attempt to factor
%p = amd(C);
disp('minimum degree ordering for symmetric matrix...')
p=symamd(C);  

toc(gensampT)

disp('Cholesky decomposition...')
[L, fail] = chol(C(p,p),'lower');
toc(gensampT)
assert(fail==0, 'Inverse covariance must be posdef');



% Generate samples
disp('generate univariate samples...')
X = randn(n,m); % "stochastic gem"
toc(gensampT)
disp('Transform samples X with cholesky decomposition of precision matrix...')
X = L' \ X;
toc(gensampT)
disp('Reorder solution...')
X(p,:) = X;
toc(gensampT)
end

%% Exeperiment with alternative methods of ordering

% method:               permutation + chol. + transform
% =========             =================================
% symamd:               2 + 8.5 + 6.2
% colperm:              0.07 + Computer locking up!
% symrcm:               2 + Computer locking up!
% amd:                  2 + 12.5 + 6.9
% colamd:               2.8 + 70 + 16
% suitesparse/camd:     2.1 + 11 + 7

