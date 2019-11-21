% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

function [global_mean_val_error, col_mean_val_error, row_mean_val_error, R_val_i,R_val_j,R_val_v] = averagingRMatrixPredictionScores(resultsLogFID,R_train,R_val,dimN,dimM)
%Compute error score

row_sums = sum(R_train,2);
col_sums = sum(R_train,1);

[R_i,R_j,R_v]=find(R_train);
[R_row_counts,~]=hist(R_i,1:dimN);
[R_col_counts,~]=hist(R_j,1:dimM);

[R_val_i,R_val_j,R_val_v]=find(R_val);
[R_val_row_counts,~]=hist(R_val_i,1:dimN);
[R_val_col_counts,~]=hist(R_val_j,1:dimM);

num_samples_trn=sum(R_row_counts);
num_samples_val=length(R_val_v);

R_row_means = row_sums./R_row_counts';
R_col_means = col_sums./R_col_counts;
R_global_mean = sum(row_sums)/num_samples_trn;

% Global mean RMSE
global_mean_train_error = sqrt(sum((R_global_mean - R_v).^2)/num_samples_trn)
global_mean_val_error = sqrt(sum((R_global_mean - R_val_v).^2)/num_samples_val)
fprintf(resultsLogFID, '\nGlobal mean RMSE \ntrn: %f\nval: %f\n',global_mean_train_error,global_mean_val_error);

% Col mean RMSE
col_mean_train_error = sqrt(sum((repelem(R_col_means,R_col_counts)' - R_v).^2)/num_samples_trn)
col_mean_val_error = sqrt(sum((repelem(R_col_means,R_val_col_counts)' - R_val_v).^2)/num_samples_val)
fprintf(resultsLogFID, '\nCol mean RMSE \ntrn: %f\nval: %f\n',col_mean_train_error,col_mean_val_error);

% Row mean RMSE
% Order values by row (not column)
[~,~,R_v_tran]=find(R_train');
[~,~,R_val_v_tran]=find(R_val');
row_mean_train_error = sqrt(sum((repelem(R_row_means,R_row_counts) - R_v_tran).^2)/num_samples_trn)
row_mean_val_error = sqrt(sum((repelem(R_row_means,R_val_row_counts) - R_val_v_tran).^2)/num_samples_val)
fprintf(resultsLogFID, '\nRow mean RMSE \ntrn: %f\nval: %f\n',row_mean_train_error,row_mean_val_error);

%[R_val_ii,R_val_jj,R_val_vv]=find(R_val);

clear R_global_mean R_row_means R_col_means row_sums col_sums R_col_counts R_row_counts 
clear R_row_counts R_col_counts R_val_row_counts R_val_col_counts

end

