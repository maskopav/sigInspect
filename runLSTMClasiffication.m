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


% Final variables for the model
Xfinal = Xfiltered;
Yfinal = Yfiltered;
signalIdsFinal = signalIdsFiltered;

% Convert labels to categorical 
Yfinal = cellfun(@(y) double(y), Yfinal, 'UniformOutput', false);


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

%%
excelFile = 'Feature_selection_results_correct.xlsx';
sheetName = 'FS';
criteriaList = {'youden', 'f1', 'recall'};

XTrainFs = [XTrain; XVal];
YTrainFs = [YTrain; YVal];

for artifactIdx=4:4
    disp(artifactIdx)

    % Data preprocessing
    [allFeatureValues, labels] = extractFeatureValues(XTrainFs, YTrainFs, artifactIdx);
    X_fs = allFeatureValues';  % Features (rows: samples, cols: features)
    Y_fs = categorical(labels'); % Labels (0 = clean, 1 = artifact)

    alpha = 1;
    classWeight_higher = computeClassWeights(Yfinal, alpha);
    alpha = 0.8;
    classWeight_mid = computeClassWeights(Yfinal, alpha);
    alpha = 0.6;
    classWeight_lower = computeClassWeights(Yfinal, alpha);
    costWeights = [classWeight_lower, classWeight_mid, classWeight_higher];
    for costIdx = 1:length(costWeights)
        costMatrix = [0 1; costWeights(costIdx) 0];
    
        for idx = 1:length(criteriaList)
            criteria = criteriaList{idx}; 
            disp(criteria)

            % Feature selection with SVM RBF kernel
            [selectedFeatures_FS, accuracy_train, sensitivity_train, specificity_train, precision_train, f1_train, svmModel] = featureSelection(X_fs, Y_fs, costMatrix, criteria);
            % Predict on unseen dataset
            [allFeatureValuesTest, labelsTest] = extractFeatureValues(XTest, YTest, artifactIdx);
            allFeatureValuesTest = allFeatureValuesTest(selectedFeatures_FS, :);

            predictions = predict(svmModel, allFeatureValuesTest');
            [accuracy_unseen, sensitivity_unseen, specificity_unseen, precision_unseen, f1_unseen] = computeEvaluationMetrics(categorical(labelsTest'), predictions);
            
            % Save Results to Excel File
            resultsTable = table(artifactIdx, ...
                strjoin(string(selectedFeatures_FS), ', '), strjoin(string(featNames(selectedFeatures_FS)), ', '), ...
                accuracy_train, sensitivity_train, specificity_train, precision_train, f1_train, ...
                accuracy_unseen, sensitivity_unseen, specificity_unseen, precision_unseen, f1_unseen, ...
                strjoin(string(costMatrix), ', '), string(criteria), ...
                'VariableNames', {'artifactIdx', 'Selected_FS_Features', 'Selected_FS_Features_Names', ...
                                  'Accuracy_Train', 'Sensitivity_Train', 'Specificity_Train', 'Precision_Train', 'F1_Score_Train', ...
                                  'Accuracy_Unseen', 'Sensitivity_Unseen', 'Specificity_Unseen', 'Precision_Unseen', 'F1_Score_Unseen', ...
                                  'Cost_Matrix', 'Criteria'});
            saveResultsToExcel(excelFile, sheetName, resultsTable);
        end
    end
end

%% Feature selection - selected features by featureSelection using sequentialfs and svm model
artifactIdx = 3;
selectedFeatures_FS = [14, 16, 20, 30, 31, 32];
Xselected = cellfun(@(x) x(selectedFeatures_FS, :), Xfiltered, 'UniformOutput', false);
Yselected = cellfun(@(y) y(artifactIdx, :), Yfiltered, 'UniformOutput', false);

% Final variables for the model
Xfinal = Xselected;
Yfinal = Yselected;
Yfinal = cellfun(@(y) categorical(double(y(:)')), Yfinal, 'UniformOutput', false);
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

%% Analysis of signals of artifact 2 POW - wierd shape of AUC curve
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

%%
% Extract second row from each cell and concatenate into a matrix
secondRowValues = cellfun(@(x) x(2, :), predictedProbs, 'UniformOutput', false);
secondRowMatrix = vertcat(secondRowValues{:});  % Convert cell array to matrix

% Compute average for each column
avgValues = mean(secondRowMatrix, 1);
avgValues = [avgValues; 0.3195    0.2148    0.1606    0.1412    0.1200    0.1082    0.0910    0.0933    0.0821    0.0732];

figure
plot(1:10,avgValues)
% Display results
% disp('Average values for each column in the second row:');
% disp(avgValues);
% 


