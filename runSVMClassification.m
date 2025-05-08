clc; clear;

sigInspectAddpath;

%% Parameters
% Define paths
dataFolder = 'data/';
csvFile = '_metadataMER2020.csv';
signalsFolder = 'signals/';

featureDataPath = fullfile(dataFolder, 'featureSetUndersampled.mat');

%% Load features suitable for model training
% LOAD FEATURE DATA (X and Y)
fprintf('Feature file exist. Loading data...\n');
load(featureDataPath, 'featureSetUndersampled');

X = featureSetUndersampled.X;        % Cell array
Y = featureSetUndersampled.Y;        % Cell array
signalIds = featureSetUndersampled.signalIds; % Cell array
featNames = featureSetUndersampled.featNames; % Cell array



%% Data split for model training
ratios = struct('train', 0.65, 'val', 0.2, 'test', 0.15);
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIds, ratios);
[trainPatientIds, trainUniquePatients] = getPatientIds(signalIds(trainIdx));
[valPatientIds, valUniquePatients] = getPatientIds(signalIds(valIdx));
[testPatientIds, testUniquePatients] = getPatientIds(signalIds(testIdx));

% Display results
fprintf('Number of training samples: %d, number of unique patients: %d\n', numel(trainIdx), trainUniquePatients);
fprintf('Number of validation samples: %d, number of unique patients: %d\n', numel(valIdx), valUniquePatients);
fprintf('Number of test samples: %d, number of unique patients: %d\n', numel(testIdx), testUniquePatients);

%% Feature selection - selected features by runFeatureSelection script using sequentialfs and svm model
artifactIdx = 4;
selectedFeatures_FS = [5, 7, 8, 13, 15, 17];
cleanToArtifactRatio = 3.25;
Xselected = cellfun(@(x) x(selectedFeatures_FS, :), X, 'UniformOutput', false);
% Yselected = cellfun(@(y) y(artifactIdx, :), Y, 'UniformOutput', false);

% Final variables for the model
Xfinal = Xselected;
Yfinal = Y;
signalIdsFinal = signalIds;

% Access the splits
XTrain = Xfinal(trainIdx, :);
YTrain = Yfinal(trainIdx, :);
XVal = Xfinal(valIdx, :);
YVal = Yfinal(valIdx, :);
XTest = Xfinal(testIdx, :);
YTest = Yfinal(testIdx, :);

% Prepare data for SVM, not in cell format, but as matrix, we dont need
% information about consecuent windows
[X_fs_train, Y_fs_train, signal_ids_train] = extractFeatureValues(XTrain, YTrain, artifactIdx, trainPatientIds);
[X_fs_val, Y_fs_val, signal_ids_val] = extractFeatureValues(XVal, YVal, artifactIdx, valPatientIds);
[X_fs_unseen, Y_fs_unseen, signal_ids_test] = extractFeatureValues(XTest, YTest, artifactIdx, testPatientIds);

%% SVM training
% Undersample at first by specified ratio
[balancedTrainX, balancedTrainY] = undersampleByRatio(X_fs_train, Y_fs_train, trainPatientIds, cleanToArtifactRatio);
svmModel = fitcsvm(balancedTrainX, balancedTrainY, ...
              'Prior', 'uniform', ...
              'KernelFunction', 'RBF', ...
              'Standardize', true);
% Convert to a model that can output probability scores
svmProbModel = fitPosterior(svmModel);

%% SVM predict
% TRAIN
[predictions, ~] = predict(svmModel, X_fs_train);
evalMetrics_train = computeEvaluationMetrics(Y_fs_train, predictions);
% To make predictions with soft labels (probabilities)
[~, predictedProbsTrain] = predict(svmProbModel, X_fs_train);
% Compute PR AUC using the function
prAUC = computePRCurveAUC(Y_fs_train, predictedProbsTrain(:,2), 1);
evalMetrics_train.prAUC = prAUC;
% Compute ROC AUC
rocAUC = computeROCAUC(Y_fs_train, predictedProbsTrain(:,2), 1);
evalMetrics_train.rocAUC = rocAUC;


% VALIDATION
predictions = predict(svmModel, X_fs_val);
evalMetrics_val = computeEvaluationMetrics(Y_fs_val, predictions);
% To make predictions with soft labels (probabilities)
[~, predictedProbsVal] = predict(svmProbModel, X_fs_val);
% Compute PR AUC using the function
prAUC = computePRCurveAUC(Y_fs_val, predictedProbsVal(:,2), 1);
evalMetrics_val.prAUC = prAUC;
% Compute ROC AUC
rocAUC = computeROCAUC(Y_fs_val, predictedProbsVal(:,2), 1);
evalMetrics_val.rocAUC = rocAUC;


% UNSEEN
predictions = predict(svmModel, X_fs_unseen);
evalMetrics_unseen = computeEvaluationMetrics(Y_fs_unseen, predictions);
% To make predictions with soft labels (probabilities)
[~, predictedProbsTest] = predict(svmProbModel, X_fs_unseen);
% Compute PR AUC using the function
prAUC = computePRCurveAUC(Y_fs_unseen, predictedProbsTest(:,2), 1);
evalMetrics_unseen.prAUC = prAUC;
% Compute ROC AUC
rocAUC = computeROCAUC(Y_fs_unseen, predictedProbsTest(:,2), 1);
evalMetrics_unseen.rocAUC = rocAUC;

disp(evalMetrics_train)
disp(evalMetrics_val)
disp(evalMetrics_unseen)

%% Save results to mat file
SVMFile = fullfile(dataFolder, 'svm_results_FREQ.mat');

results_svm.svmProbModel = svmProbModel;
results_svm.svmModel = svmModel;
results_svm.evalMetrics_train = evalMetrics_train;
results_svm.evalMetrics_val = evalMetrics_val;
results_svm.evalMetrics_unseen = evalMetrics_unseen;
results_svm.predictedProbsTrain = predictedProbsTrain;
results_svm.predictedProbsVal = predictedProbsVal;
results_svm.predictedProbsTest = predictedProbsTest;

results_svm.XTrain = X_fs_train;
results_svm.YTrain = Y_fs_train;
results_svm.XVal   = X_fs_val;
results_svm.YVal   = Y_fs_val;
results_svm.XTest  = X_fs_unseen;
results_svm.YTest  = Y_fs_unseen;
results_svm.featNames = featNames(selectedFeatures_FS);

save(SVMFile, 'results_svm');