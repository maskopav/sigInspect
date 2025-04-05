clc; clear;

sigInspectAddpath;

% Parameters
% Define paths
dataFolder = 'data/';
csvFile = '_metadataMER2020.csv';
signalsFolder = 'signals/';

loadedSignalsPath = fullfile(dataFolder, 'loadedSignals.mat');
featureDataPath = fullfile(dataFolder, 'featureDataMerged.mat');

% Load or create data and features suitable for model training
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


%%% Remove NANs 
% Nan values can be removed based on raw signal data or feature data
validIdx = findValidIndices(signalData);
signalData = signalData(validIdx);
annotationsData = annotationsData(validIdx);
X = X(validIdx);
Y = Y(validIdx);
signalIds = signalIds(validIdx);

%%% Convert annotations from numeric format to binary format
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


%%% Data split for model training
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

%
excelFile = 'FS_results.xlsx';
sheetName = 'FS';
% Youden and recall similar results -> only Youden and F1 score
criteriaList = {'youden', 'f1'};

XTrainFs = [XTrain; XVal];
YTrainFs = [YTrain; YVal];

%%% artifactIdx
%1   'clean'    'CLN'
%2    'power'    'POW'
%3    'baseline'    'BASE'
%4    'frequency artifact'    'FREQ'
%5    'irritated neuron'    'IRIT'
%6    'other'    'OTHR'
%7    'artifact'    'ARTIF'

for artifactIdx=2:4
    disp(artifactIdx)

    % Data preprocessing
    [X_fs, Y_fs] = extractFeatureValues(XTrainFs, YTrainFs, artifactIdx);

    % Subset for testing
    % X_fs = X_fs(1:1000,:);
    % Y_fs = Y_fs(1:1000);

    % Class weights - three different weights 
    YTrainFsArtif = cellfun(@(y) y(artifactIdx, :), YTrainFs, 'UniformOutput',false);
    alpha = 1;
    classWeight_higher = computeClassWeights(YTrainFsArtif, alpha);
    alpha = 0.8;
    classWeight_mid = computeClassWeights(YTrainFsArtif, alpha);
    alpha = 0.6;
    classWeight_lower = computeClassWeights(YTrainFsArtif, alpha);

    costWeights = [classWeight_lower, classWeight_mid, classWeight_higher];
    for costIdx = 1:length(costWeights)
        costMatrix = [0 1; costWeights(costIdx) 0];
    
        for idx = 1:length(criteriaList)
            criteria = criteriaList{idx}; 
            disp(['Evaluating using criterion: ', criteria])

            % Feature selection with SVM RBF kernel
            [selectedFeatures_FS, accuracy_train, sensitivity_train, specificity_train, precision_train, f1_train, svmModel] = featureSelection(X_fs, Y_fs, costMatrix, criteria);
            % Predict on unseen dataset
            [X_fs_unseen, Y_fs_unseen] = extractFeatureValues(XTest, YTest, artifactIdx);
            X_fs_unseen = X_fs_unseen(:, selectedFeatures_FS);

            predictions = predict(svmModel, X_fs_unseen);
            [accuracy_unseen, sensitivity_unseen, specificity_unseen, precision_unseen, f1_unseen] = computeEvaluationMetrics(Y_fs_unseen, predictions);
            
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