% Parameters
method = 'sgd'; % 'gd' or 'sgd'
sigma_r = 0.4;  % Variance of entries
D=10;           % Latent dimension
eta = 0.003;    % Learning rate
gamma=0.1;      % Parameter for graph kernel

load data.mat
% Positive entries are valid entries, construct mask matricies
mask_train = ones(size(trainSet)) & trainSet;
mask_test = ones(size(testSet)) & testSet;
mask_val = ones(size(valSet)) & valSet;

% Scale entries to be in [0,1]
trainSet = trainSet/5.0;
testSet = testSet/5.0;
valSet = valSet/5.0;

[N,M] = size(trainSet);

% Get covariance matrix for columns/movies (assuming diagonal)
K_v = 0.2 * eye(M,M);
K_v_inv = inv(K_v);

% Get covariance matrix for rows/users based on the graph kernel (using
% side information)
K_u = graphKernel(Graph, gamma); 
K_u_inv = pinv(K_u);

% Start training
if strcmp(method, 'gd')
    [U, V, vRMSE,time] = kpmf_gd(trainSet, mask_train, D, K_u_inv, K_v_inv, sigma_r, eta, valSet, mask_val);
elseif strcmp(method, 'sgd')
    [U, V, vRMSE,time] = kpmf_sgd(trainSet, mask_train, D, K_u_inv, K_v_inv, sigma_r, eta, valSet, mask_val);
end

% Compute RMSE on the test set
rmse=sqrt(sum(sum(((U*V'-testSet).*mask_test).^2))/sum(sum(mask_test)))

