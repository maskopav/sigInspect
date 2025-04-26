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
signalIds = featureSetUndersampled.signalIds; % Vector
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
selectedFeatures_FS = [6, 9, 14, 16];
Xselected = cellfun(@(x) x(selectedFeatures_FS, :), X, 'UniformOutput', false);
Yselected = cellfun(@(y) y(artifactIdx, :), Y, 'UniformOutput', false);

% Final variables for the model
Xfinal = Xselected;
Yfinal = Yselected;
Yfinal = cellfun(@(y) categorical(double(y(:)')), Yfinal, 'UniformOutput', false);
signalIdsFinal = signalIds;

% Access the splits
XTrain = Xfinal(trainIdx, :);
YTrain = Yfinal(trainIdx, :);
XVal = Xfinal(valIdx, :);
YVal = Yfinal(valIdx, :);
XTest = Xfinal(testIdx, :);
YTest = Yfinal(testIdx, :);

%% Classes in datasets 
trainLabels = cellfun(@(x) x', YTrain, 'UniformOutput', false);
trainLabels = vertcat(trainLabels{:});
fprintf('Training dataset\n');
tabulate(trainLabels)

valLabels = cellfun(@(x) x', YVal, 'UniformOutput', false);
valLabels = vertcat(valLabels{:});
fprintf('Validation dataset\n');
tabulate(valLabels)

testLabels = cellfun(@(x) x', YTest, 'UniformOutput', false);
testLabels = vertcat(testLabels{:});
fprintf('Test dataset\n');
tabulate(testLabels)
%% LSTM
% Parameters for LSTM network
mode = 'binary'; % or 'binary'

% Handle number of classes and class weights
if strcmp(mode, 'binary')
    alpha = 0.5;
    classWeight = computeClassWeights(Yfinal, alpha);
    % classWeight =  3.0134;
    classWeights = [1, classWeight]; % Only for binary classification
    numClasses = 2;
elseif strcmp(mode, 'multi')
    numClasses = maxN-1;
    %classWeights = %ones(1, numClasses); 
end

lstmSettings = struct();

inputSize = size(XTrain{1}, 1);   % Number of features

lstmSettings.lstmUnits = 5;                    % Number of LSTM units
lstmSettings.dropOut = 0.3;
lstmSettings.maxEpochs = 30;                    % Number of epochs
lstmSettings.miniBatchSize = 16;                % Mini-batch size
lstmSettings.initialLearnRate = 0.0001;          % Initial learning rate
lstmSettings.validationFrequency = 10;          % Frequency of validation checks
lstmSettings.validationPatience = 8;            % Early stopping if no improvement for 5 epochs
lstmSettings.classWeights = classWeights;

% Call function to train the model and predict
[net, predictedProbsTrain, predictedProbsVal, predictedProbsTest] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, lstmSettings, mode);
%% Classify unseen data and save results to excel file 
excelFile = 'results/lstm_classification/FS_results_undersampling_cost.xlsx';
sheetName = 'LSTM';


evalMetricsVal = evaluateModel(predictedProbsVal, YVal, mode, artifactIdx);
evalMetricsTrain = evaluateModel(predictedProbsTrain, YTrain, mode, artifactIdx, evalMetricsVal.optimalThreshold);
evalMetricsTest = evaluateModel(predictedProbsTest, YTest, mode, artifactIdx, evalMetricsVal.optimalThreshold);

saveLSTMResultsToExcel(artifactIdx, selectedFeatures_FS, evalMetricsTrain, evalMetricsVal, evalMetricsTest, lstmSettings, excelFile, sheetName)

%% Soft label plots
fig = figure;
plotSoftLabelDistributions(predictedProbsTest, YTest);
%savePlotToExcel(fig, excelFile, 'TestSoftLabelsPlot', 'plot_test.png');

% plotSoftLabelDistributions(predictedProbsTest, YTest);


%% Analysis of signals in case of wierd shape of AUC curve
% Windows corresponding to a False Positive Rate (FPR) between 0.1 and 0.2 
positiveClassIdx = 2;  % Assuming second column corresponds to the positive class
probsPos = cellfun(@(x) x(positiveClassIdx, :)', predictedProbsTest, 'UniformOutput', false);
probs = vertcat(probsPos{:});


%%
labels = cellfun(@(x) x', YTest, 'UniformOutput', false);
labels = vertcat(labels{:});
labels = double(string(labels));
% Compute ROC & AUC
[fpr, tpr, thresholds, auc] = perfcurve(labels, probs, 1);

% Identify threshold range for FPR between 0.1 and 0.2
fpr_range = (fpr >= 0.1) & (fpr <= 0.2);
threshold_range = thresholds(fpr_range);

% Get min and max threshold in this range
minThreshold = min(threshold_range);
maxThreshold = max(threshold_range);

disp(['Threshold range for FPR 0.1-0.2: [', num2str(minThreshold), ', ', num2str(maxThreshold), ']']);

% Find windows with predicted probabilities in the target range
matchingWindows = cellfun(@(x) ((x >= minThreshold) & (x <= maxThreshold))', probsPos, 'UniformOutput', false);

% Find indices where at least one window in a signal meets the condition
matchingSignalIndices = find(cellfun(@(x) any(x), matchingWindows));

%%
idx = 4; % matchingSignalIndices(30);
signalTest = signalData(testIdx, :);
signal = signalTest{idx};
fs = 24000;
signalProbs = probsPos{idx}';
signalLabels = double(string(YTest{idx}));
signalArtifNames = 'POW';
useSpectrogram = true;

visualizeSignalWithPredictions(signal, fs, signalProbs, signalLabels, signalArtifNames, useSpectrogram)

%% Plot artifact distributions
counts = plotArtifactHistogram(X, Y, signalIds);

for j = 1:numel(counts)
    sortedCounts = counts{j}.artifactCounts;
    sortedPatients = counts{j}.patientIds; 
    totalCount = sum(sortedCounts);
    percent = 100 * sortedCounts / totalCount;

    fprintf('\n=== Artifact Type %d ===\n', j);
    fprintf('%-12s %-10s %-10s\n', 'PatientID', 'Count', 'Percent');

    for k = 1:numel(sortedPatients)
        fprintf('%-12s %-10d %-9.2f%%\n', sortedPatients{k}, sortedCounts(k), percent(k));
    end
end
