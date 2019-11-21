% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

% This is the main demo code for GPMF 
% NOTE: Reruires installation of GRALS code https://github.com/rofuyu/exp-grmf-nips15
%  The code is already packed, just requires running the install file in
%  the grmf directory.

%% Initialise libraries
    addpath('MovieLens/ml-100k');          % https://grouplens.org/datasets/movielens/
    addpath('grmf/grmf-core/matlab');       % https://github.com/rofuyu/exp-grmf-nips15
    % Comment our MATLAB path and add octave path for use with octave
    %addpath('grmf/grmf-core/octave');       % https://github.com/rofuyu/exp-grmf-nips15
    addpath('kpmf');                        % https://people.eecs.berkeley.edu/~tinghuiz/
    addpath('gpmf_Mstep');                  % M-step graph update functions

% For deterministic reruns 
    rng(1);

% Logging (GRALS logging is printed to the MATLAB terminal)
    resultsLogFID = fopen('GPMF_exp_movielens100k_results_output.log','w');
    fprintf(resultsLogFID, '\nNew experiment on MovieLens 100K data %s .... \n',datetime('now'));

% Shared model params
    dimD = 10; % Number of latent features

%% Import and process data from MovieLens

% Import covariate data and create kNN graphs
[G_U,G_V,dimN,dimM] = createGraphsFromMovieLensCovariateData();
% Create Laplacian matrices
L_U = laplacian(graph(G_U));
L_V = laplacian(graph(G_V));

% Import user rating data
    U1Base = tdfread('u1.base','|'); % Take one CV split for demo
    U1Test = tdfread('u1.test','|');
    % {i,j,r} tuples as sparse matrix representation 
    R_train=U1Base.user_id_item_id_rating_timestamp(:,1:3);
    R_val=U1Test.user_id_item_id_rating_timestamp(:,1:3);
    % Normalise data (optional)
    R_train(:,3)=R_train(:,3)./5; 
    R_val(:,3)=R_val(:,3)./5;
    
    R_train=sparse(R_train(:,1),R_train(:,2),R_train(:,3),dimN,dimM);
    R_val=sparse(R_val(:,1),R_val(:,2),R_val(:,3),dimN,dimM);


%% Simple benchmark scores
[global_mean_val_error,col_mean_val_error,row_mean_val_error,R_val_i,R_val_j,R_val_v]=averagingRMatrixPredictionScores(resultsLogFID,R_train,R_val,dimN,dimM);

%% PMF - no graph (pausing to retrieve initial values for GPMF, then continuing)
PMF_Comp_Time=tic;

CG_iter=3;
max_algo_iters=100;
Gral_config = ['-k ',num2str(dimD),' -e 0.001 -t ',num2str(max_algo_iters),' -g ',num2str(CG_iter)];
Lap_diag_gamma_U=1.2;
Lap_diag_gamma_V=1.2;
gamma_U_diag=Lap_diag_gamma_U*speye(dimN);
gamma_V_diag=Lap_diag_gamma_V*speye(dimM);

[U_PMF,V_PMF,RMSE_PMF,WallTime_PMF] = glr_mf_train(R_train,R_val,gamma_U_diag,gamma_V_diag,Gral_config);
U_PMF = U_PMF';
V_PMF = V_PMF';

PMFcompT=toc(PMF_Comp_Time)

PMF_val_error = sqrt( (length(R_val_v)^-1) * sum((sum(U_PMF(R_val_i,:) .* V_PMF(R_val_j,:),2) - R_val_v).^2) );
disp(['PMF RMSE: ',num2str(PMF_val_error)]);
fprintf(resultsLogFID, '\n===\nPMF, Laplacian (gamma_U=%f,gamma_V=%f) \nval: %f (%.2f secs.)\n',...
    Lap_diag_gamma_U,Lap_diag_gamma_V,PMF_val_error,PMFcompT);
fprintf(resultsLogFID, '\nSettings: D=%i, CG iter=%i\n===\n\n',dimD,CG_iter);

%% GRALS - full graph
GRALS_Comp_Time=tic;

CG_iter=2;
max_algo_iters=200;
gamma_U=0.0001;
gamma_V=0.0001;
gamma_L=0.4;
Gral_config = ['-k ',num2str(dimD),' -e 0.001 -t ',num2str(max_algo_iters),' -g ',num2str(CG_iter)];

% regularise Laplacian matrices (Section 2, Laplacian Matrix of GPMF paper, and Equation 5 of GRALS paper)
L_U_reg = gamma_L*L_U + gamma_U*speye(dimN);
L_V_reg = gamma_L*L_V + gamma_V*speye(dimM);
    
[U_GRALS,V_GRALS,RMSE_GRALS,WallTime_GRALS] = glr_mf_train(R_train,R_val,L_U_reg,L_V_reg,Gral_config);
U_GRALS = U_GRALS';
V_GRALS = V_GRALS';
GRALSCompT=toc(GRALS_Comp_Time);

GRALS_val_error = sqrt( (length(R_val_v)^-1)*sum((sum(U_GRALS(R_val_i,:) .* V_GRALS(R_val_j,:),2) - R_val_v).^2) );

disp(['GRALS RMSE: ',num2str(GRALS_val_error)]);
fprintf(resultsLogFID, '\n===\nGRALS, reg. Laplacian (gamma_L=%f,gamma_U=%f,gamma_V=%f) \nval: %f (%.2f secs.)\n',...
    gamma_L,gamma_U,gamma_V,GRALS_val_error,GRALSCompT);
fprintf(resultsLogFID, '\nSettings: D=%i, CG iter=%i\n===\n\n',dimD,CG_iter);

%% KPMF - full graph (sgd optimisation)
KPMF_Comp_Time=tic;

sig2=0.2;  % observation noise
eta = 0.00001;  % Learning rate
gamma=1.1;  % Kernel hyper-parameter

% Prepare data for KPMF code
trainSet = full(R_train);
valSet = full(R_val);
mask_train = ones(size(trainSet)) & trainSet;
mask_val = ones(size(valSet)) & valSet;

K_u = graphKernel(G_U, gamma); 
K_u_inv = pinv(K_u);
K_v = graphKernel(G_V, gamma); 
K_v_inv = pinv(K_v);

[U_KPMF,V_KPMF,RMSE_KPMF,time_KPMF] = kpmf_sgd(trainSet, mask_train, dimD, K_u_inv, K_v_inv, sig2, eta, valSet, mask_val);
KPMFCompT=toc(KPMF_Comp_Time);

KPMF_val_error = sqrt( (length(R_val_v)^-1)*sum((sum(U_KPMF(R_val_i,:) .* V_KPMF(R_val_j,:),2) - R_val_v).^2) );
disp(['KPMF RMSE: ',num2str(KPMF_val_error)]);
fprintf(resultsLogFID, '\n===\nKPMF, Kernel (sigma^2=%f,gamma=%f)\nval: %f (%.2f secs.)\n',...
    sig2,gamma,KPMF_val_error,KPMFCompT);
fprintf(resultsLogFID, '\nSettings: D=%i\n===\n\n',dimD);

%% GPMF - Use stored PMF values for initialisation, run M-step, then E-step once
GPMF_Comp_Time=tic;

% Initialise U,V with no graph (PMF)
GPMF_Comp_Time_init=tic;

CG_iter=3;
max_algo_iters=35;
Gral_config = ['-k ',num2str(dimD),' -e 0.001 -t ',num2str(max_algo_iters),' -g ',num2str(CG_iter)];
Lap_diag_gamma=1.2;
gamma_U_diag=Lap_diag_gamma*speye(dimN);
gamma_V_diag=Lap_diag_gamma*speye(dimM);

[U_GPMF_INIT,V_GPMF_INIT,RMSE_GPMF_INIT,WallTime_GPMF_INIT] = glr_mf_train(R_train,R_val,gamma_U_diag,gamma_V_diag,Gral_config);
U_GPMF_INIT = U_GPMF_INIT';
V_GPMF_INIT = V_GPMF_INIT';

GPMFCompTinit=toc(GPMF_Comp_Time_init);

%% M-step: Approximate posterior covariance with diagonalized row-wise covariance
GPMF_Comp_Time_MApproxPost_Full=tic;

sig2=0.05;
alph=sig2^-1;

GPMF_Comp_Time_MApproxPost_U=tic;
SPostHat_U_ds = EstimatePosteriorCovarianceForMStep(V_GPMF_INIT,sparse(dimN,dimN),dimN,R_val_j,R_val_i,dimD,alph,L_U);
GPMFCompTMApproxPost_U=toc(GPMF_Comp_Time_MApproxPost_U);

GPMF_Comp_Time_MApproxPost_V=tic;
SPostHat_V_ds = EstimatePosteriorCovarianceForMStep(U_GPMF_INIT,sparse(dimM,dimM),dimM,R_val_i,R_val_j,dimD,alph,L_V);
GPMFCompTMApproxPost_V=toc(GPMF_Comp_Time_MApproxPost_V);

GPMFCompTMApproxPost_Full=toc(GPMF_Comp_Time_MApproxPost_Full);

disp('Approximate posterior covariance estimates done.')

%% M-step: Compute outer produce of MAP estimates of latent faetures 

%(only compute outer products matching non-zeros in the adjacency matrix,
% as significantly less to compute).

GPMF_Comp_Time_MOuterProductMAP_Full=tic;

U_GPMF_INIT_c = U_GPMF_INIT' - mean(U_GPMF_INIT,2)';  % zero-mean the data
V_GPMF_INIT_c = V_GPMF_INIT' - mean(V_GPMF_INIT,2)';

GPMF_Comp_Time_MOuterProductMAP_U=tic;
S_sparse_U = SparseCovariance(U_GPMF_INIT_c,G_U);  % Embarrassingly parallel potential for extremely large dimension
GPMFCompTMOuterProductMAP_U=toc(GPMF_Comp_Time_MOuterProductMAP_U);

GPMF_Comp_Time_MOuterProductMAP_V=tic;
S_sparse_V = SparseCovariance(V_GPMF_INIT_c,G_V);
GPMFCompTMOuterProductMAP_V=toc(GPMF_Comp_Time_MOuterProductMAP_V);

% M-step expected sample covariance
% GPMF paper - Equation 10 
Expected_sample_cov_U = SPostHat_U_ds + S_sparse_U;
Expected_sample_cov_V = SPostHat_V_ds + S_sparse_V;

%clear SPostHat_U_ds S_sparse_U SPostHat_V_ds S_sparse_V

%% M-step thresholding of expected sample covariance to approximate GLASSO
% solution sparsity structure

GPMF_Comp_Time_Threshold_Full=tic;

tau_U=0.0; % Graph sparsity inducing strength - increase for even sparser solution
tau_V=0.0; 

% Sparse Inverse cov est. 
% ... for U
GPMF_Comp_Time_Threshold_U=tic;

Cov_thr = CovThresholdWithSparsityConstraints(Expected_sample_cov_U,tau_U,U_GPMF_INIT_c);
L_new_U = LaplacianFromMatrixStruct(Cov_thr);
if isempty(L_new_U)
    L_new_U = inv(Cov_thr);
end
Sigma_new_U = Cov_thr;
G_U_thresh = GraphFromMatrixStruct(Cov_thr);

GPMFCompTThreshold_U=toc(GPMF_Comp_Time_Threshold_U);

% ... for V
GPMF_Comp_Time_Threshold_V=tic;

Cov_thr = CovThresholdWithSparsityConstraints(Expected_sample_cov_V,tau_V,V_GPMF_INIT_c);
L_new_V = LaplacianFromMatrixStruct(Cov_thr);
if isempty(L_new_V)
    L_new_V = inv(Cov_thr);
end
Sigma_hat_V = Cov_thr;
G_V_thresh = GraphFromMatrixStruct(Cov_thr);

GPMFCompTThreshold_V=toc(GPMF_Comp_Time_Threshold_V);

GPMFCompTThreshold_Full=toc(GPMF_Comp_Time_Threshold_Full);

% Graph properties to analyse the sparse estimation reduction
L_U_sparsity = nnz(L_U)/(dimN^2);
L_U_new_sparsity = nnz(L_new_U)/(size(L_new_U,1)^2);
L_V_sparsity = nnz(L_V)/(size(L_V,1)^2)
L_V_new_sparsity = (nnz(L_new_V) + dimN)/(size(L_new_V,1)^2);

%% E-step: MAP estimate of U,V with updated graph
GPMF_Comp_Time_E_step=tic;

%CG_iter=2;
CG_iter=2;
max_algo_iters=200;
gamma_U = 0.0001;
gamma_V = 0.0001;
gamma_L=0.4;

Gral_config = ['-k ',num2str(dimD),' -e 0.01 -t ',num2str(max_algo_iters),' -g ',num2str(CG_iter)];

L_U_reg = gamma_L*L_new_U + gamma_U*speye(dimN);
L_V_reg = gamma_L*L_new_V + gamma_V*speye(dimM);

[U_grals_GPMF_graph,V_grals_GPMF_graph,RMSE_GPMF,WallTime_GPMF] = glr_mf_train(R_train,R_val,L_U_reg,L_V_reg,Gral_config);
U_grals_GPMF_graph = U_grals_GPMF_graph';
V_grals_GPMF_graph = V_grals_GPMF_graph';

GPMFCompTEstep=toc(GPMF_Comp_Time_E_step);
GPMFCompT=toc(GPMF_Comp_Time);

GPMF_val_error = sqrt( (length(R_val_v)^-1)*sum((sum(U_grals_GPMF_graph(R_val_i,:) .* V_grals_GPMF_graph(R_val_j,:),2) - R_val_v).^2) );

disp(['GPMF RMSE: ',num2str(GPMF_val_error)]);
fprintf(resultsLogFID, '\n===\nGPMF, reg. Laplacian (gamma_L=%f,gamma_U=%f,gamma_V=%f) \nval: %f (E-Step: %.2f, Full: %.2f secs.)\n',...
    gamma_L,gamma_U,gamma_V,GPMF_val_error,GPMFCompTEstep,GPMFCompT);
fprintf(resultsLogFID, '\nSettings: D=%i, CG iter=%i\n===\n\n',dimD,CG_iter);


%% KPMF A+ - GRAEM updated graph (sgd optimisation)
KPMF_GRAEM_Comp_Time=tic;

sig2=0.1;  % observation noise
eta = 0.00001;  % Learning rate
gamma=0.5;  % Kernel hyper-parameter

% Prepare data for KPMF code
trainSet = full(R_train);
valSet = full(R_val);
mask_train = ones(size(trainSet)) & trainSet;
mask_val = ones(size(valSet)) & valSet;

K_u = graphKernel(G_U_thresh, gamma); 
K_u_inv = pinv(K_u);
K_v = graphKernel(G_V_thresh, gamma); 
K_v_inv = pinv(K_v);

[U_KPMF_GRAEM,V_KPMF_GRAEM,RMSE_KPMF_GRAEM,time_KPMF] = kpmf_sgd(trainSet, mask_train, dimD, K_u_inv, K_v_inv, sig2, eta, valSet, mask_val);
KPMFGRAEMCompT=toc(KPMF_GRAEM_Comp_Time);

KPMF_GRAEM_val_error = sqrt( (length(R_val_v)^-1)*sum((sum(U_KPMF_GRAEM(R_val_i,:) .* V_KPMF_GRAEM(R_val_j,:),2) - R_val_v).^2) );
disp(['KPMF (GRAEM A+) RMSE: ',num2str(KPMF_GRAEM_val_error)]);
fprintf(resultsLogFID, '\n===\nKPMF GRAEM A+ graphs, Kernel (sigma^2=%f,gamma=%f)\nval: %f (%.2f secs.)\n',...
    sig2,gamma,KPMF_GRAEM_val_error,KPMFCompT);
fprintf(resultsLogFID, '\nSettings: D=%i\n===\n\n',dimD);

%% Summary
fprintf(1,'\n ==== SUMMARY: \n');

fprintf(1,'\n Data matrix R size %i x %i, training observations %i, validation observations % i  \n\n',...
    size(R_train,1),size(R_train,2),nnz(R_train),nnz(R_val));

fprintf(1,'Global Mean \t\tRMSE: %f \n',global_mean_val_error);
fprintf(1,'Col. Mean \t\tRMSE: %f \n',col_mean_val_error);
fprintf(1,'Row. Mean \t\tRMSE: %f \n',row_mean_val_error);
fprintf(1,'PMF \t\t\tRMSE: %f \t\t(%f secs.)\n',PMF_val_error,PMFcompT);

fprintf(1,'\n Graph side information: \n\tusers:\t %i x %i with %i edges\n\tmovies:\t %i x %i with %i edges \n',...
    size(G_U,1),size(G_U,1),nnz(G_U),size(G_V,1),size(G_V,1),nnz(G_V));

fprintf(1,'GRALS \t\t\tRMSE: %f \t\t(%f secs.)\n',GRALS_val_error,GRALSCompT);
fprintf(1,'KPMF \t\t\tRMSE: %f \t\t(%f secs.)\n',KPMF_val_error,KPMFCompT);
fprintf(1,'GPMF \t\t\tRMSE: %f \t\t(%f secs.)\t\t (proportion of edges remaining: users %f, movies %f) \n',...
    GPMF_val_error,GPMFCompT,nnz(G_U_thresh) / nnz(G_U),nnz(G_V_thresh) / nnz(G_V));
fprintf(1,'KPMF (GRAEM A+) \tRMSE: %f \t\t(%f secs.)\n',KPMF_GRAEM_val_error,KPMFGRAEMCompT);




