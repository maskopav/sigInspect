clc; clear;

sigInspectAddpath;

%% Parameters
% Define paths
dataFolder = 'data/';
csvFile = '_metadataMER2020.csv';
signalsFolder = 'signals/';

loadedSignalsPath = fullfile(dataFolder, 'loadedSignals.mat');
featureDataPath = fullfile(dataFolder, 'featureDataMerged.mat');

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

%% Data split for model training
ratios = struct('train', 0.6, 'val', 0.2, 'test', 0.2);
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIdsFiltered, ratios);

% Display results
fprintf('Number of training samples: %d\n', numel(trainIdx));
fprintf('Number of validation samples: %d\n', numel(valIdx));
fprintf('Number of test samples: %d\n', numel(testIdx));


%% Feature selection - selected features by runFeatureSelection script using sequentialfs and svm model
artifactIdx = 3;
selectedFeatures_FS = [14, 16, 20, 30, 31, 32];
Xselected = cellfun(@(x) x(selectedFeatures_FS, :), Xfiltered, 'UniformOutput', false);
Yselected = cellfun(@(y) y(artifactIdx, :), Yfiltered, 'UniformOutput', false);

% Final variables for the model
Xfinal = Xselected;
Yfinal = Yselected;
Yfinal = cellfun(@(y) categorical(double(y(:)')), Yfinal, 'UniformOutput', false);
signalIdsFinal = signalIdsFiltered;

% Access the splits
XTrain = Xfinal(trainIdx, :);
YTrain = Yfinal(trainIdx, :);
XVal = Xfinal(valIdx, :);
YVal = Yfinal(valIdx, :);
XTest = Xfinal(testIdx, :);
YTest = Yfinal(testIdx, :);


%% LSTM
% Parameters for LSTM network
mode = 'binary'; % or 'binary'

% Handle number of classes and class weights
if strcmp(mode, 'binary')
    alpha = 0.6;
    classWeight = computeClassWeights(Yfinal, alpha)
    classWeights = [1, classWeight]; % Only for binary classification
    numClasses = 2;
elseif strcmp(mode, 'multi')
    numClasses = maxN-1;
    %classWeights = %ones(1, numClasses); 
end

inputSize = size(XTrain{1}, 1);   % Number of features
lstmUnits = 20;                    % Number of LSTM units
dropOut = 0.3;
maxEpochs = 30;                    % Number of epochs
miniBatchSize = 32;                % Mini-batch size
initialLearnRate = 0.0001;          % Initial learning rate
validationFrequency = 10;          % Frequency of validation checks
validationPatience = 5;            % Early stopping if no improvement for 5 epochs


% Call function to train the model and predict
[net, predictedProbs] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, classWeights, ...
    lstmUnits, dropOut, maxEpochs, miniBatchSize, ...
    initialLearnRate, validationFrequency, validationPatience, mode);
%% Classify unseen data
[accuracy, sensitivity, specificity, auc, optimalThreshold] = evaluateModel(predictedProbs, YTest, mode)

%% Analysis of signals in case of wierd shape of AUC curve
% Windows corresponding to a False Positive Rate (FPR) between 0.1 and 0.2 
positiveClassIdx = 2;  % Assuming second column corresponds to the positive class
probsPos = cellfun(@(x) x(positiveClassIdx, :)', predictedProbs, 'UniformOutput', false);
probs = vertcat(probsPos{:});

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
idx = matchingSignalIndices(7);
signalTest = signalData(testIdx, :);
signal = signalTest{idx};
fs = 24000;
signalProbs = probsPos{idx}';
signalLabels = YTest{idx};
signalArtifNames = 'POW';
useSpectrogram = true;

visualizeSignalWithPredictions(signal, fs, signalProbs, signalLabels, signalArtifNames, useSpectrogram)

%% Soft label analysis
allProbs = cellfun(@(x) x(2, :)', predictedProbs, 'UniformOutput', false);
allProbs = vertcat(allProbs{:});

% HISTOGRAM OF SOFT LABELS
figure;
histogram(allProbs, 20)
xlabel('Soft label probability'); ylabel('Count');
title(['Distribution of soft labels, artif: ', num2str(artifactIdx)])

% AVG SOFT LABEL PER WINDOW
allProbsWindows = cellfun(@(x) x(2, :), predictedProbs, 'UniformOutput', false);
allProbsWindows = vertcat(allProbsWindows{:}); 

% Get true labels (0 = clean, 1 = artifact)
allTrueLabels = cellfun(@(x) x(:)', YTest, 'UniformOutput', false);
allTrueLabels = vertcat(allTrueLabels{:}); 
allTrueLabels = double(string(allTrueLabels));

[numSamples, numWindows] = size(allProbsWindows);
[winX, ~] = meshgrid(1:numWindows, 1:numSamples);
xData = winX(:);
yData = allProbsWindows(:);
labels = allTrueLabels(:);  % 0 = clean, 1 = artifact

% Colors
cmap = [0 0 1; 1 0 0];  % blue = clean, red = artifact
colorData = cmap(labels + 1, :);  % label+1 for indexing

% Split data by label for curves
probsCleanMask = allProbsWindows;
probsCleanMask(allTrueLabels == 1) = NaN;
probsArtifMask = allProbsWindows;
probsArtifMask(allTrueLabels == 0) = NaN;

avgClean = mean(probsCleanMask, 1, 'omitnan');
stdClean = std(probsCleanMask, 0, 1, 'omitnan');
avgArtifact = mean(probsArtifMask, 1, 'omitnan');
stdArtifact = std(probsArtifMask, 0, 1, 'omitnan');

% Plotting
figure; hold on;

% Plot shaded error bars for both groups
shadedErrorBar(1:numWindows, avgClean, stdClean, 'lineProps', {'b', 'DisplayName', 'Clean avg ± STD'});
shadedErrorBar(1:numWindows, avgArtifact, stdArtifact, 'lineProps', {'r', 'DisplayName', 'Artifact avg ± STD'});

% Scatter points
scatter(xData(labels==0), yData(labels==0), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Clean samples');
scatter(xData(labels==1), yData(labels==1), 10, 'r', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Artifact samples');
hold off

% Labels and legend
xlabel('Windows'); ylabel('Soft label');
ylim([0 1])
title('Soft labels per window');
legend('Location', 'best');
grid on;



