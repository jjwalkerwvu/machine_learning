% iris_script.m

% This reads in the iris dataset and performs logistic regression

% use csvread

%data_arr=csvread('bezdekIris.data')
myfile = fopen ("bezdekIris.data", "r");
% read in some comma-separated data, where there is a string at the end
data_arr=textscan(myfile,'%f%f%f%f%s','Delimiter',',');
% close the file_in_loadpath
fclose(myfile);

% trim off the last row?
data_arr{1}(end)=[];
data_arr{2}(end)=[];
data_arr{3}(end)=[];
data_arr{4}(end)=[];
data_arr{5}(end)=[];

% set up the cases using strcmp on the last cell index, the flower names (classes)

y_class=strcmp(data_arr{5}(:),'Iris-setosa')+... 
  2*strcmp(data_arr{5}(:),'Iris-versicolor')+... 
  3*strcmp(data_arr{5}(:),'Iris-virginica');
  
  
% should randomize and create train, cv, and test sets;
% for now, just train the data

% random indices that we will need for later
rand_inds=randperm(length(data_arr{1}));

X_feat=[data_arr{1} data_arr{2} data_arr{3} data_arr{4}];

num_labels=3;

lambda=0;

% 
[all_theta] = oneVsAll(X_feat, y_class, num_labels, lambda);

% Now see how well the model (regression coefficients) predict the data
p = predictOneVsAll(all_theta, X_feat);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y_class)) * 100);