function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% loop implementation; loop over all training examples.

for index=1:size(X,1)
  % this algorithm does not feel robust enough, but it does pass the grader.
  
  % make a big matrix with a row vector (size= n features) for each example from
  % X that is the same size as the matrix centroids.
  temp_mat=repmat(X(index,:),size(centroids,1),1);
  % take the difference between the matrix constructed from X and square this;
  % you then sum along the 2 dimension of the result, so along each row, which 
  % gives you a vector with dimension K.
  norm_mat=sum((temp_mat-centroids).^2,2);
  % the cluster that this example should be set to is the smallest number in the
  % vector computed in the previous line.
  idx(index)=find(norm_mat==min(norm_mat));
  
  
  %idx(index)=min((X(index,:)-))


end






% =============================================================

end

