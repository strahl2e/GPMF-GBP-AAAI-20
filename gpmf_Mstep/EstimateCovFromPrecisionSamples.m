% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

function [S_hat] = EstimateCovFromPrecisionSamples(L,n,p)
% An initial estimate of the inverse of the prior (laplacian) graph
%   Use efficient sampling from a multivariate to generate samples from
%   prior, then use sample covariance as an estimate of inverse of prior.
%estCovFromPrecSamples=tic

    estCovFromPrecSamplesT=tic;
    disp('EstimatePostCov')
    toc(estCovFromPrecSamplesT)
    disp('genSamples...')
    genSamples = genSamp(L, n)';
    toc(estCovFromPrecSamplesT)
    % Compute only non-zero covariances for efficiency
    disp('Find upper triangular of L...')
    [Gi_triu, Gj_triu] = find(triu(L,1));
    toc(estCovFromPrecSamplesT)
    disp('Reorgainse samples...')
    initSamps_c = genSamples - mean(genSamples);
    sigma_hat = var(initSamps_c,1);
    init_is = initSamps_c(:,Gi_triu);
    init_js = initSamps_c(:,Gj_triu);
    toc(estCovFromPrecSamplesT)
    disp('Compute inner products to estimate covariance...')
    Sigma_hat = sparse(Gi_triu, Gj_triu, sum(init_is.*init_js,1),p,p);
    toc(estCovFromPrecSamplesT)
    disp('Sum prior and likelihood approximation for posterior approximation...')
    S_hat = Sigma_hat + Sigma_hat' + sparse(1:p, 1:p,sigma_hat);
    toc(estCovFromPrecSamplesT)

end

