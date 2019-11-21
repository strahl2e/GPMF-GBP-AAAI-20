% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.


function [S_post_hat] = EstimatePosteriorCovarianceForMStep(X,S,p,X_is,beta_is,D,alph,L)
% Estimate the posterior covariance for the M-step of GPMF
% GPMF paper, Equation .. to compute each C_d 
% The covariance of the columns is dropped by only taking the diagonal of
% the evidence X

EstPostCovT=tic;

FULL=0;
KS=1; %Computer kronecker sum true=1, false=0
SparCons=1;
SparConsAvg=0;

% If some row/column has no observations.


%size(X,2);
disp('Square data elements...');
X2 = X.^2;
toc(EstPostCovT)
if SparConsAvg
    % Instead of taking D lots of N x N samples we just average over the
    % D lots of N x N diagonals and take one sample. D times faster!
    %C_bar_D_sum_MN = zeros(p,1);
    disp('Sum d columns of data..');
    X2_D_summed = sum(X2,2);
    toc(EstPostCovT)
    size(X2_D_summed)
    %disp('Collect X^2 summed over d elements into sparse matrix..');
    %X2_reps = sparse(beta_is,X_is,X2_D_summed(X_is));
    %size(X2_reps)
    %toc(EstPostCovT)
    %disp('Sum the observations of X^2 summed over d...');
    %C_bar_D_sum_MN = sum(X2_reps,2);
    %toc(EstPostCovT)
    disp('find unique elements and counts...');
    [a,~,c]=unique(beta_is);
    toc(EstPostCovT)
    disp('sum observations using accumarray...');
    C_bar_D_sum_MN = accumarray(c,X2_D_summed(X_is));
    toc(EstPostCovT)
    disp('Divide by D...');
    C_bar_D = C_bar_D_sum_MN./D;
    toc(EstPostCovT)
    disp('Compute L...');
    L_D = L + alph*spdiags(C_bar_D,0,p,p);
    toc(EstPostCovT)
    disp('EstimateCovFromPrecSamp...');
    S_D = EstimateCovFromPrecisionSamples(L_D,2*D,p);
    toc(EstPostCovT)
    S_post_hat = S_D;
elseif SparCons
    % Use linear cholesky decomp sampling as oppose to quadtratic sparse inverse
    % subset sampling.
    %C_bar_d_MN = zeros(p,D);
    %C_bar_d_MN = zeros(p,1);
    S_ds = sparse(p,p);
    disp('find unique elements and counts...');
    [a,~,c]=unique(beta_is);
    zeroObsIndx = setdiff(1:p,a)
    toc(EstPostCovT)
    for d=1:D
        disp('Compute C_d...');
        d
        toc(EstPostCovT)
        disp('sum observations using accumarray...');
        C_bar_d_MN = accumarray(c,X2(X_is,d));
        toc(EstPostCovT)
        if ~isempty(zeroObsIndx)
            C_bar_d_MN_padded = zeros(1,p);
            C_bar_d_MN_padded(a) = C_bar_d_MN;
            C_bar_d_MN = C_bar_d_MN_padded';
        end
        %X2_reps = sparse(beta_is,X_is,X2(X_is,d));
        %toc(EstPostCovT)
        %C_bar_d_MN = sum(X2_reps,2);
%         for ind=1:p    
%             ind
%             p
%             disp('compute C_bar')
%             toc(sparsconcomp)
%             C_bar_d_MN(ind) = sum(X2(X_is(beta_is==ind),d));
%             disp('done')
%             toc(sparsconcomp)
%         end
        disp('Compute L_d...');
        L_d = L + alph*spdiags(C_bar_d_MN,0,p,p);
        toc(EstPostCovT)
        disp('Estimate Post. cov. from samples...'); 
        S_d = EstimateCovFromPrecisionSamples(L_d,2*D,p);
        toc(EstPostCovT)
        S_ds = S_ds + S_d;
    end    
    S_post_hat = S_d/D;
else
    if ~KS %Approximate with Matrix Normal Kronecker Produce inverse
        C_bar_d_MN = zeros(D,1);
        for d=1:D
            curr_diag=zeros(p,1);
            for ind=1:p
                curr_diag(ind) = sum(X2(X_is(beta_is==ind),d))^-1;
            end
            C_bar_d_MN(d) = sum(curr_diag)/p;
        end
        S_post_hat = alph*(sum(C_bar_d_MN)/D)*S;
        %figure;imagesc(SPost_hat);colorbar;
    else % Posterior computation with Kronecker sum inverse
        if FULL
            C_bar_d_MN = sparse(p*D,p*D);
            for d=1:D
                for dd=1:D
                    for ind=1:p
                        C_bar_d_MN((d-1)*p + ind,(dd-1)*p + ind) = sum(X(X_is(beta_is==ind),d).*X(X_is(beta_is==ind),dd));
                    end
                end
            end
            SPost_hat_full = sparseinv(kron(speye(D),L) + alph*C_bar_d_MN);
            S_post_hat=sparse(p,p);
            for ind=1:D
                start_of_block=(ind-1)*p + 1;
                end_of_block=ind*p;
                S_post_hat = S_post_hat + SPost_hat_full(start_of_block:end_of_block,start_of_block:end_of_block);
            end
            S_post_hat = S_post_hat/D;        
        else
            C_bar_d_MN = zeros(p,D);
            for d=1:D
                for ind=1:p    
                    C_bar_d_MN(ind,d) = sum(X2(X_is(beta_is==ind),d));
                end
            end
            %spdiags(reshape([[1,2];[3,4];[5,6]],2*3,1),0,6,6)
            % SPost_hat_full = inv(kron(speye(D),L) + alph*spdiags(reshape(C_bar_d_MN,D*p,1),0,D*p,D*p));
            SPost_hat_full = sparseinv(kron(speye(D),L) + alph*spdiags(reshape(C_bar_d_MN,D*p,1),0,D*p,D*p));
            S_post_hat=sparse(p,p);
            for ind=1:D
                start_of_block=(ind-1)*p + 1;
                end_of_block=ind*p;
                S_post_hat = S_post_hat + SPost_hat_full(start_of_block:end_of_block,start_of_block:end_of_block);
            end
            S_post_hat = S_post_hat/D;        
        end
        %figure;imagesc(SPost_hat);colorbar;
    end
end
%i=1:p;
%C_js_MN = (sum(reshape(sum(repmat(X2(X_is,:),1,length(i)).*sparse(kron(((beta_is == i)),ones(1,n)))),p,n))/p).^-1;
%SPost_hat = alph*(sum(C_js_MN)/n).*S_Post_hat_t_m_1;
end

% i=1:dimN;
% j=1:dimM;
% V_grals_2 = V_grals.^2;
% U_grals_2 = U_grals.^2;
% C_js_MN_U = (sum(reshape(sum(repmat(V_grals_2(R_train(:,2),:),1,length(i)).*sparse(kron(((R_train(:,1) == i)),ones(1,dimD)))),dimN,dimD))/dimN).^-1;
% C_js_MN_V = (sum(reshape(sum(repmat(U_grals_2(R_train(:,1),:),1,length(j)).*sparse(kron(((R_train(:,2) == j)),ones(1,dimD)))),dimM,dimD))/dimM).^-1;
% SPostHat_U_ds = (sum(C_js_MN_U)/dimD).*Sigma_hat_0_U;
% SPostHat_V_ds = (sum(C_js_MN_V)/dimD).*Sigma_hat_0_V;
