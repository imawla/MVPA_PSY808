%% load data obtained from 01_roi_extract
% this script has been adopted from Thad Polk
% University of Michigan
% PSY 808

if exist('roi_vals_per_vox_per_subj.txt', 'file') == 2
    % read file for roi values per voxel per subject
    % rows - subjects, columns - voxels
    data=single(dlmread('roi_vals_per_vox_per_subj.txt'));
elseif exist('roi_vals_per_vox_per_subj.txt', 'file') == 0
    error('roi_vals_per_vox_per_subj.txt not present...run 01_roi_extract first');
end

if exist('labellist.txt', 'file') == 2
    % read file for roi values per voxel per subject
    % rows - subjects, columns - voxels
    CorrectLabels=(dlmread('labellist.txt')');
elseif exist('labellist.txt', 'file') == 0
    error('labellist.txt not present...run 01_roi_extract first');
end


%% k-Nearest Neighbor classification with N-fold cross validation

k = 1; % How many neighbors for kNN classifier (usually odd)
Nfolds = 12;  % How many folds (divisions of data) for cross-validation
indices = crossvalind('Kfold', size(data,1), Nfolds);

% Randomly divide data into folds. indices is a vector where each entry
% is an integer between 1 and Nfolds, indicating which fold each data
% vector belongs to

cp = classperf(CorrectLabels); % Initialize a classifer performance object

for i = 1:Nfolds
    test = (indices == i); train = ~test;
    % test indicates data to be tested in this fold. train indicates data
    % used in training for this fold.
    classes= knnclassify(data(test,:),data(train,:),CorrectLabels(train), k);
    % Perform kNN classification on this fold's test data based on training data
    classperf(cp,classes,test);
    % Update the CP object with the classification results from this fold
end

cp.CorrectRate % Output the average correct classification rate
 
%% SVM classification with N-fold cross validation

Nfolds = 12;  % How many folds (divisions of data) for cross-validation
indices = crossvalind('Kfold', size(data,1), Nfolds);

% Randomly divide data into folds. indices is a vector where each entry
% is an integer between 1 and Nfolds, indicating which fold each datapoint
% belongs to

cp = classperf(CorrectLabels); % Initialize a classifer performance object

for i = 1:Nfolds
    test = (indices == i); train = ~test;
    % test indicates data to be tested in this fold. train indicates data
    % used in training for this fold.
    svmStruct = svmtrain(data(train,:),CorrectLabels(train));
    % Train a support vector machine on this fold's training data
    classes = svmclassify(svmStruct,data(test,:));
    % Use the trained SVM to classify this fold's test data
    classperf(cp,classes,test);
    % Update the CP object with the classification results from this fold
end

cp.CorrectRate % Output the average correct classification rate
