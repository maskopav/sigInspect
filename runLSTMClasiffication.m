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
    [signalData, annotationsData, signalIds] = extractSignalData(loadedSignals);
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


%% Remove NANs 
% Nan values can be removed based on raw signal data or feature data
validIdx = findValidIndices(signalData);
signalData = signalData(validIdx);
annotationsData = annotationsData(validIdx);
X = X(validIdx);
Y = Y(validIdx);
signalIds = signalIds(validIdx);

%% Convert annotations from numeric format to binary format
% multiclass or binary class option -> choose correct mode

mode = 'multi'; % or 'binary'

% Number of artifact types for type mode only
maxN = 6;
%0    'clean'    'CLN'
%1    'power'    'POW'
%2    'baseline'    'BASE'
%4    'frequency artifact'    'FREQ'
%8    'irritated neuron'    'IRIT'
%16    'other'    'OTHR'
%32    'artifact'    'ARTIF'

Yconverted = convertToBinaryLabels(Y, mode, maxN);

% Add clean class as the first row
Yconverted = cellfun(@(y) [~any(y, 1); y], Yconverted, 'UniformOutput', false);

% Remove signals which contains sixth type of artifacts ('ARTIF'), which is not suitable for multiclass classification
if strcmp(mode, 'multi')
    % Find indices of signals containing artifact type 6 ('OTHR') +
    % artifact type 7 ('ARTIF')
    artifactTypeToDeleteIdx = cellfun(@(x) (any(x(6, :)) || any(x(7, :))), Yconverted);
    
    Xfiltered = X(~artifactTypeToDeleteIdx);
    Yfiltered = Yconverted(~artifactTypeToDeleteIdx);
    signalIdsFiltered = signalIds(~artifactTypeToDeleteIdx);

    % Remove the 6th and 7th row
    Yfiltered = cellfun(@(y) y(1:5, :), Yfiltered, 'UniformOutput', false);

    % Display number of removed signals
    disp(['Removed ', num2str(sum(artifactTypeToDeleteIdx)), ' signals containing unwanted artifact types']);
end


% Final variables for the model
Xfinal = Xfiltered;
Yfinal = Yfiltered;
signalIdsFinal = signalIdsFiltered;

% Convert labels to categorical 
Yfinal = cellfun(@(y) double(y), Yfinal, 'UniformOutput', false);

%% Compute class weights
alpha = 0.8; % Increase this to put more emphasis on rare classes
classWeights = computeClassWeights(Yfinal, alpha);

classWeights(end) = 0.6
%% Data split for model training
ratios = struct('train', 0.6, 'val', 0.2, 'test', 0.2);
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIdsFiltered, ratios);

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

% Handle number of classes and class weights
if strcmp(mode, 'binary')
    numClasses = 2;
    classWeights = [0.6, 1]; % Only for binary classification
elseif strcmp(mode, 'multi')
    numClasses = maxN-1;
    %classWeights = %ones(1, numClasses); 
end

inputSize = size(XTrain{1}, 1);   % Number of features
lstmUnits = 20;                    % Number of LSTM units
dropOut = 0.5;
maxEpochs = 30;                    % Number of epochs
miniBatchSize = 32;                % Mini-batch size
initialLearnRate = 0.001;          % Initial learning rate
validationFrequency = 10;          % Frequency of validation checks
validationPatience = 5;            % Early stopping if no improvement for 5 epochs

disp(['Size of first YTrain sample: ', num2str(size(YTrain{1}))]);
disp(['Expected size: [', num2str(numClasses), ', sequenceLength]']);

% Call function to train the model and predict
[net, predictedProbs] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, classWeights, ...
    lstmUnits, dropOut, maxEpochs, miniBatchSize, ...
    initialLearnRate, validationFrequency, validationPatience, mode);
%% Classify unseen data

[accuracy, sensitivity, specificity, auc, optimalThreshold] = evaluateModel(predictedProbs, YTest, mode)
