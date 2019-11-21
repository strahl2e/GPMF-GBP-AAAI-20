function [U, V,RMSE,time] = kpmf_sgd(R, mask, D, K_u_inv, K_v_inv, sigma_r, eta, R2, mask2)
%
% Author: Hanhuai Shan
%
% KPMF with stochastic gradient descent
%
% N:        num of rows in R
% M:        num of cols in R
% D:        latent dimensions
% 
% Input:
%   R:          N*M, matrix for training
%   mask:       N*M, indicator matrix, 1 indicates a valid entry
%   D:          latent dimensions
%   K_u_inv:    N*N, precision matrix on rows
%   K_v_inv:    M*M, precision matrix on columns
%   sigma_r:    scaler, variance for the univariate Gaussian to generate each entry
%               in R
%   eta:        scaler, learning rate
%   R2:         N*M, validate matrix
%   mask2:      N*M, indicator matrix for the validation set
%
% Output:
%   U:          N*D, latent matrix on rows
%   V:          M*D, latent matrix on columns
%   RMSE:       scaler, RMSE on validation set
%   time:       running time
%----------------------------------------------------------------------

tic;

N = size(R, 1);     % #rows
M = size(R, 2);     % #columns

numEachRow=sum(mask,2);
numEachCol=sum(mask,1)';


%**************************************************************************

% Initialize U and V

Rsvd=R.*mask+sum(sum(R.*mask))/sum(sum(mask))*(1-mask);
[U,S,V]=svds(Rsvd,D);
U=U*sqrt(S);
V=V*sqrt(S);
clear Rsvd;

rs=sqrt(sum(sum((R2-U*V').^2.*mask2))/sum(sum(mask2)));

%**************************************************************************

% Start learning
maxepoch=100;
minepoch=5;
epsilon=0.00001;

[rowind,colind,value]=find(R.*mask);
train_vec=[rowind,colind,value];
len=length(value);
clear rowind colind value;
batch_num=12;
batch_size=round(len/batch_num);
momentum=0.2;

U_inc=zeros(size(U));
V_inc=zeros(size(V));
for epoch=1:maxepoch
    train_vec=train_vec(randperm(len),:);
    for batch=1:batch_num
        if batch<batch_num
            brow   = double(train_vec((batch-1)*batch_size+1:batch*batch_size,1));
            bcol   = double(train_vec((batch-1)*batch_size+1:batch*batch_size,2));
            bval   = double(train_vec((batch-1)*batch_size+1:batch*batch_size,3));
        elseif batch==batch_num
            brow   = double(train_vec((batch-1)*batch_size+1:len,1));
            bcol   = double(train_vec((batch-1)*batch_size+1:len,2));
            bval   = double(train_vec((batch-1)*batch_size+1:len,3));
        end
        
        pred = sum(U(brow,:).*V(bcol,:),2);
        K_u_invU=(U'*K_u_inv)';
        diag_K_u_inv_U=repmat(diag(K_u_inv),1,D).*U;
        K_v_invV=(V'*K_v_inv)';
        diag_K_v_inv_V=repmat(diag(K_v_inv),1,D).*V;
        gd_u=-2/(sigma_r^2)*(repmat((bval-pred),1,D).*V(bcol,:))+K_u_invU(brow,:)./repmat(numEachRow(brow),1,D)+diag_K_u_inv_U(brow,:)./repmat(numEachRow(brow),1,D);
        gd_v=-2/(sigma_r^2)*(repmat((bval-pred),1,D).*U(brow,:))+K_v_invV(bcol,:)./repmat(numEachCol(bcol),1,D)+diag_K_v_inv_V(bcol,:)./repmat(numEachCol(bcol),1,D);
        
        
        dU=zeros(N,D);
        dV=zeros(M,D);
         
        for b=1:length(brow);
            r=brow(b);
            c=bcol(b);
            dU(r,:)=dU(r,:)+gd_u(b,:);
            dV(c,:)=dV(c,:)+gd_v(b,:);
        end

        U_inc=U_inc*momentum+dU*eta;
        V_inc=V_inc*momentum+dV*eta;
        
        U=U-U_inc;
        V=V-V_inc;
    end
    
    RMSE(epoch)=sqrt(sum(sum((R2-U*V').^2.*mask2))/sum(sum(mask2)));
    
    disp(['epoch' ,int2str(epoch),', validation rmse=',num2str(RMSE(epoch))]);
    
    if (epoch > minepoch  && (RMSE(epoch-1) - RMSE(epoch))/RMSE(epoch-1)<epsilon)
        break;
    end
    
end

time=toc;

