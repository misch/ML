function [C, A] = Kmeans_MicheleWyss(X, Cinit)
    % input:
    %   X: the data matrix, each column represents a data point
    %   Cinit: initial cluster centers, each column represents a cluster center
    % output:
    %   C: the cluster centers after the k-means, the format is the same as
    %   Cinit
    %   A: row vector of labels, A(i)=j means that X(:,i) belongs to the
    %   cluster j, ie. C(:,j) is the closest cluster center

    % number of clusters
    k = size(Cinit, 2);

    % transpose the stuff for usage of dsearchn(...)
    Cinit = transpose(Cinit);
    X = transpose(X);

    % Matlab has already a nearest point search implemented:
    % Citation from mathworks documentation:
    % Y = dsearchn(X,XI) returns the indices Y of the closest points in X for
    % each point in XI. X is an m-by-n matrix representing m points in n-dimensional 
    % space. XI is a p-by-n matrix, representing p points in n-dimensional space. 
    % The output Y is a column vector of length p. 
    Y = zeros(size(X, 2), 1);
    YNew = dsearchn(Cinit, X); 

    % repeat until the clusters don't change anymore
    while(~isequal(Y,YNew))
        % initial cluster centroids
        C = zeros([k, size(Cinit, 2)]);

        % new cluster centroids:
        % i-th centroid = mean of the data points having label i
        for i = 1:k 
            C(i,:) = sum(X(YNew==i, :))/sum(YNew==i);
        end

        Y = YNew;
        YNew = dsearchn(C, X); 
     end

    A = Y';
    C = C';
end

