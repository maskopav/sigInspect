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

%% Soft label plots
plotSoftLabelDistributions(results_lstm.predictedProbsTest, results_lstm.YTest);
%savePlotToExcel(fig, excelFile, 'TestSoftLabelsPlot', 'plot_test.png');

% plotSoftLabelDistributions(predictedProbsTest, YTest);

%% Visualize atlas of signals with soft labels
load('data/lstm_results_POW.mat', 'results_lstm')
loadedSignalsPath = fullfile(dataFolder, 'loadedSignals.mat');
load(loadedSignalsPath, 'loadedSignals');
[signalData, annotationsData, signalIdsOrig] = extractSignalData(loadedSignals);

%%
[trueLabels, predictedLabels, signalIdsLabels] = extractFeatureValues(results_lstm.YTest, results_lstm.predictedProbsTest, 2, signalIds(testIdx));
% Match signalIdsLabels (from windows) to signalIdsOrig (from original signals)
matchedIdx = cellfun(@(id) find(strcmp(id, signalIdsOrig)), signalIdsLabels);

% Define soft label bins
binEdges = [0, 0.15, 0.3, 0.45, 0.6];
binLabels = {'0-0.25', '0.25-0.5', '0.5-0.75', '0.75-1'};

% Discretize predicted labels into bins
binIdx = discretize(predictedLabels, binEdges);

% Pre-allocate
numBins = numel(binLabels);
binnedWindows = cell(numBins, 1); % Each cell contains list of windows in that bin
samplingFreq = 24000; % Hz
windowLength = 1; % seconds

windowsPerSignal = 10;

for b = 1:numBins
    % Indices of windows in bin b
    winIdx = find(binIdx == b);
    
    % Extract windows
    binnedWindows{b} = cell(numel(winIdx), 1);
    for j = 1:numel(winIdx)
        i = winIdx(j);
        
        % Compute which signal and window within signal
        signalIdx = ceil(i / windowsPerSignal);
        windowNum = mod(i-1, windowsPerSignal) + 1;
        
        % Get corresponding signal and divide into windows
        signal = signalData{signalIdx};
        %samplingFreq = samplingFreqs(signalIdx);
        windows = divideIntoWindows(signal, windowLength, samplingFreq);
        
        % Extract the correct window
        binnedWindows{b}{j} = windows(:, :, windowNum);
    end
end


figure;
for b = 1:numBins
    subplot(2,2,b);
    hold on;
    
    % Plot windows in bin b
    for j = 21:23 %numel(binnedWindows{b})
        window = binnedWindows{b}{j}; % [channels x segmentLength]
        % Plot
        plot(window);
    end
    title(sprintf('Soft label %s', binLabels{b}));
    xlabel('Samples');
    ylabel('Amplitude');
    ylim([-300 300])
end



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
