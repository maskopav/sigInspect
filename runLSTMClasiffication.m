clc; clear;

sigInspectAddpath;

%% Parameters
% Define paths
dataFolder = 'data/';
csvFile = '_metadataMER2020.csv';
signalsFolder = 'signals/';

loadedSignalsPath = fullfile(dataFolder, 'loadedSignals.mat');
featureDataPath = fullfile(dataFolder, 'featureData.mat');

%% Load or create data and features suitable for model training
% LOAD OR CREATE SIGNALS AND ANNOTATIONS
if isfile(loadedSignalsPath)
    fprintf('Loaded signals file exists. Loading data...\n');
    load(loadedSignalsPath, 'loadedSignals');
else
    fprintf('Loaded signals file not found. Running `loadSignalsAndAnnotations`...\n');
    loadedSignals = loadSignalsAndAnnotations(dataFolder, csvFile, signalsFolder, true, loadedSignalsPath);
end

% LOAD OR CREATE FEATURE DATA (X and Y)
if isfile(featureDataPath)
    fprintf('Feature file exist. Loading data...\n');
    load(featureDataPath, 'featureSet');

    X = featureSet.X;        % Cell array
    Y = featureSet.Y;        % Cell array
    signalIds = featureSet.signalIds; % Vector
    featNames = featureSet.featNames; % Cell array

else
    fprintf('Feature files not found. Extracting features...\n');
    % Extract signal and annotation data
    [signalCellData, annotationsData, signalIds] = extractSignalData(loadedSignals);

    % Define feature computation parameters
    featNames = {'pow', 'sigP90', 'sigP95', 'sigP99', 'ksnorm', 'powDiff', 'maxCorr', ...
                 'maxNormPSD', 'stdNormPSD', 'psdP75', 'psdP90', 'psdP95', 'psdP99', ...
                 'maxAbsDiffPSD', 'psdF100', 'psdBase', 'psdPow', 'psdFreq', 'psdMaxStep'};
    samplingFreq = 24000; % Hz
    windowLength = 1; % seconds

    % Compute features
    [X, Y, featNames] = computeFeaturesForLSTM(signalCellData, annotationsData, windowLength, samplingFreq, featNames);

    % Save feature data
    featureSet = struct('X', {X}, ...
                     'Y', {Y}, ...
                     'featNames', {featNames}, ...
                     'signalIds', {signalIds});
    save(featureDataPath, 'featureSet', '-v7.3');
    fprintf('Features data saved to %s\n', featureDataPath);
end


%% Remove NANs, converting labels
% Remove sequences with NaN values
[X, Y, signalIds] = removeNaNSequences(X, Y, signalIds);

% Convert multiclass labels to binary
Ybinary = convertToBinaryLabels(Y);

% Final variables for the model
Xfinal = X;
Yfinal = Ybinary;

%% Data split for model training
ratios = struct('train', 0.6, 'val', 0.2, 'test', 0.2);
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIds, ratios);

% Access the splits
XTrain = Xfinal(trainIdx, :);
YTrain = Yfinal(trainIdx, :);
XVal = Xfinal(valIdx, :);
YVal = Yfinal(valIdx, :);
XTest = Xfinal(testIdx, :);
YTest = Yfinal(testIdx, :);

% Display results
fprintf('Number of training samples: %d\n', numel(trainIdx));
fprintf('Number of validation samples: %d\n', numel(valIdx));
fprintf('Number of test samples: %d\n', numel(testIdx));


%% LSTM

% Parameters for LSTM network
inputSize = size(XTrain{1}, 1);   % Number of features
numClasses = 2;                    % Binary classification
classWeights = [0.6, 1];           % Class weights (e.g., for imbalanced data)
lstmUnits = 20;                    % Number of LSTM units
dropOut = 0.5;
maxEpochs = 30;                    % Number of epochs
miniBatchSize = 32;                % Mini-batch size
initialLearnRate = 0.001;          % Initial learning rate
validationFrequency = 10;          % Frequency of validation checks
validationPatience = 5;            % Early stopping if no improvement for 5 epochs

% Call function to train the model and predict
[net, predictedProbs] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, classWeights, lstmUnits, dropOut, maxEpochs, miniBatchSize, ...
    initialLearnRate, validationFrequency, validationPatience);
%% Classify unseen data
labels = cellfun(@(x) x', YTest, 'UniformOutput', false);
labels = vertcat(labels{:});

[accuracy, sensitivity, specificity, auc, optimalThreshold] = evaluateModel(predictedProbs, labels);
