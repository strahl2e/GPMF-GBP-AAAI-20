% Graph-based prior probabilistic matrix factorisation (GPMF) demo
% MovieLens 100k data https://grouplens.org/datasets/movielens/
% 
% Author: Jonathan Strahl 
% 
% URL: https://github.com/strahl2e/GPMF-GBP-AAAI-20
% Date: Nov 2019
% Ref: Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

function [G_U,G_V,dimN,dimM] = createGraphsFromMovieLensCovariateData()
%CREATE GRAPH FROM COVARIATE SIDE INFO 
% 
% Load covariate side information for users and movies from the MovieLens
% data.  Create binary vectors from the features, then run kNN (k=10) to
% produce the adjacency matrix. More details on processing data in Rao, 
% 2015, NIPS paper.

    % Import MovieLens user and item features
    UUser = tdfread("u.user",'|');  % User features
    UItem = tdfread("u.item",'|');  % Item features
    dimN = size(UUser.user_id,1);
    dimM = size(UItem.movie_id,1);

    % Create G_U, similarity matrix for users
    % Start with 22 dimensional feature vectors
    UUserOccupation = cellstr(UUser.occupation);
    UserOcupationList = {'administrator','artist','doctor','educator','engineer',...
        'entertainment','executive','healthcare','homemaker','lawyer','librarian',...
        'marketing','none','programmer','retired','salesman','scientist',...
        'student','technician','writer'};
    % ,'other','none' removed 
    User_feature_vec = [UUser.user_id,UUser.age,UUser.gender == 'F',zeros(length(UUser.user_id),length(UserOcupationList))];
    for i=1:length(UserOcupationList)
        User_feature_vec(:,i+3) = strcmp(UserOcupationList{i},UUserOccupation);
    end
    % kNN (k=10), Euclidean distance metric
    [knn_idx knn_D] = knnsearch(User_feature_vec(:,2:end),User_feature_vec(:,2:end),'K',11,'IncludeTies',true);
    final_knn = zeros(dimN,10);
    % Solve situation of more than ten equally distant by randomly sampling
    for i =1:dimN
        self_loop_idx = find(knn_idx{i} == i);
        knn_idx{i}(self_loop_idx)=[];
        knn_D{i}(self_loop_idx)=[];
        cut_off_dist = knn_D{i}(10);
        below_cut_off_idx = find(knn_D{i} < cut_off_dist);
        on_cut_off_idx = find(knn_D{i}==cut_off_dist);
        on_cut_off_rand_selected = randperm(length(on_cut_off_idx),10-length(below_cut_off_idx));
        final_knn(i,:) = knn_idx{i}([below_cut_off_idx,on_cut_off_rand_selected]);
    end
    user_graph_tuple=[repelem(1:size(final_knn,1),10)',reshape(final_knn',size(final_knn,1)*size(final_knn,2),1),ones(size(final_knn,1)*10,1)];
    G_U_directed = sparse(user_graph_tuple(:,1),user_graph_tuple(:,2),user_graph_tuple(:,3),dimN,dimN);
    G_U = G_U_directed + G_U_directed';

    % Create G_V, similarity matrix for users
    UItemFieldNames = fieldnames(UItem);
    fn=UItemFieldNames(1);
    Movie_feature_vec=zeros(length(UItem.(fn{1})),numel(UItemFieldNames(7:end)));
    MovieFeatureFieldNames=UItemFieldNames([1,7:end]);
    for i=1:length(MovieFeatureFieldNames)
        fn=MovieFeatureFieldNames(i);
        Movie_feature_vec(:,i)=UItem.(fn{1});
    end
    % kNN, k=10, Euclidean distance
    [knn_idx knn_D] = knnsearch(Movie_feature_vec(:,2:end),Movie_feature_vec(:,2:end),'K',11,'IncludeTies',true);
    final_knn = zeros(dimM,10);
    for i=1:dimM
        self_loop_idx = find(knn_idx{i} == i);
        knn_idx{i}(self_loop_idx)=[];
        knn_D{i}(self_loop_idx)=[];
        cut_off_dist = knn_D{i}(10);
        below_cut_off_idx = find(knn_D{i} < cut_off_dist);
        on_cut_off_idx = find(knn_D{i}==cut_off_dist);
        on_cut_off_rand_selected = randperm(length(on_cut_off_idx),10-length(below_cut_off_idx));
        final_knn(i,:) = knn_idx{i}([below_cut_off_idx,on_cut_off_rand_selected]);
    end
    movie_graph_tuple=[repelem(1:size(final_knn,1),10)',reshape(final_knn',size(final_knn,1)*size(final_knn,2),1),ones(size(final_knn,1)*10,1)];
    G_V_directed = sparse(movie_graph_tuple(:,1),movie_graph_tuple(:,2),movie_graph_tuple(:,3),dimM,dimM);
    G_V = G_V_directed + G_V_directed';
end

