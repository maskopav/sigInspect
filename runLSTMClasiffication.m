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
Xselected = cellfun(@(x) x(selectedFeatures_FS, :), X, 'UniformOutput', false);
Yselected = cellfun(@(y) categorical(double(y(artifactIdx, :))), Y, 'UniformOutput', false);

% Final variables for the model
Xfinal = Xselected;
Yfinal = Yselected;
% Yfinal = cellfun(@(y) categorical(double(y(:)')), Yfinal, 'UniformOutput', false);
signalIdsFinal = signalIds;

% Access the splits
XTrain = Xfinal(trainIdx, :);
YTrain = Yfinal(trainIdx, :);
XVal = Xfinal(valIdx, :);
YVal = Yfinal(valIdx, :);
XTest = Xfinal(testIdx, :);
YTest = Yfinal(testIdx, :);

%% To resolve error in LSTM training for artifact 4
% Error using trainNetwork
% The order of the class names of layer 5 must match the order of the class names of the validation data.
%To get the class names of the validation data, use the categories function.

% Swap first two items in XVal
tempX = XVal{1};
XVal{1} = XVal{2};
XVal{2} = tempX;

% Swap first two items in YVal
tempY = YVal{1};
YVal{1} = YVal{2};
YVal{2} = tempY;


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
    % classWeight =  2;
    % classWeight = 4;
    classWeights = [1, classWeight]; % Only for binary classification
    numClasses = 2;
elseif strcmp(mode, 'multi')
    numClasses = maxN-1;
    %classWeights = %ones(1, numClasses); 
end

lstmSettings = struct();

inputSize = size(XTrain{1}, 1);   % Number of features

lstmSettings.lstmUnits = 8;                    % Number of LSTM units
lstmSettings.dropOut = 0.4;
lstmSettings.maxEpochs = 30;                    % Number of epochs
lstmSettings.miniBatchSize = 16;                % Mini-batch size
lstmSettings.initialLearnRate = 0.0005;          % Initial learning rate
lstmSettings.validationFrequency = 8;          % Frequency of validation checks
lstmSettings.validationPatience = 10;            % Early stopping if no improvement for 5 epochs
lstmSettings.classWeights = classWeights;

% Call function to train the model and predict
[net, predictedProbsTrain, predictedProbsVal, predictedProbsTest] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, lstmSettings, mode);

%% Classify unseen data and save results to excel file 
excelFile = 'results/lstm_classification/LSTM_results.xlsx';
sheetName = 'LSTM';

evalMetricsVal = evaluateModel(predictedProbsVal, YVal, mode, artifactIdx);
evalMetricsTrain = evaluateModel(predictedProbsTrain, YTrain, mode, artifactIdx, evalMetricsVal.optimalThreshold);
evalMetricsTest = evaluateModel(predictedProbsTest, YTest, mode, artifactIdx, evalMetricsVal.optimalThreshold);

saveLSTMResultsToExcel(artifactIdx, selectedFeatures_FS, evalMetricsTrain, evalMetricsVal, evalMetricsTest, lstmSettings, excelFile, sheetName)

%% Save final LSTM model
% Save everything into a struct
results_lstm.net = net;
results_lstm.predictedProbsTrain = predictedProbsTrain;
results_lstm.predictedProbsVal = predictedProbsVal;
results_lstm.predictedProbsTest = predictedProbsTest;

results_lstm.XTrain = XTrain;
results_lstm.YTrain = YTrain;
results_lstm.XVal   = XVal;
results_lstm.YVal   = YVal;
results_lstm.XTest  = XTest;
results_lstm.YTest  = YTest;
results_lstm.featNames = featNames;

% Optional: save the struct to a .mat file for later reuse
save('data/lstm_results_POW.mat', 'results_lstm');
