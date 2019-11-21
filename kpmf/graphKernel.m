function K = graphKernel(G, gamma)
%
% Author: Tinghui Zhou
%
% Compute the covariance matrix based on the regularized graph laplacian kernel 
%
% N:    num of nodes in the graph
% 
% Input:
%   G:              N*N, undirected graph adjacency matrix
%   beta:           scaler, parameter used in graph kernel
%   kernel_str:     string, kernel name
%
% Output:
%   K:              N*N, covariance matrix
%--------------------------------------------------------

N = size(G,1);
K = zeros(N,N);

% The degree matrix
Deg = diag(sum(G));

% The graph laplacian
L = Deg - G;

K = inv((eye(N,N) + gamma * L));