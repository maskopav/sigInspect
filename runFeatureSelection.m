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
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIdsFiltered, ratios);
[trainPatientIds, trainUniquePatients] = getPatientIds(signalIdsFiltered(trainIdx));
[valPatientIds, valUniquePatients] = getPatientIds(signalIdsFiltered(valIdx));
[testPatientIds, testUniquePatients] = getPatientIds(signalIdsFiltered(testIdx));

% Display results
fprintf('Number of training samples: %d, number of unique patients: %d\n', numel(trainIdx), trainUniquePatients);
fprintf('Number of validation samples: %d, number of unique patients: %d\n', numel(valIdx), valUniquePatients);
fprintf('Number of test samples: %d, number of unique patients: %d\n', numel(testIdx), testUniquePatients);

%%
excelFile = 'FS_results_undersampling_cost.xlsx';
sheetName = 'FS_cost';
% Youden and recall similar results -> only Youden and F1 score
criteriaList = {'youden'};


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
    [X_fs_train, Y_fs_train, signal_ids_train] = extractFeatureValues(XTrain, YTrain, artifactIdx, trainPatientIds);

    % Subset for testing
    % X_fs_train = X_fs_train(1:1000,:);
    % Y_fs_train = Y_fs_train(1:1000);
    % signal_ids_train = signal_ids_train(1:1000);

    % Class weights - three different weights 
    YFsArtif = cellfun(@(y) y(artifactIdx, :), YTrain, 'UniformOutput',false);

    cleanToArtifactRatios = [1.5, 2, 2.5, 3, 3.5, 4];
    for cleanToArtifactIdx= 1:length(cleanToArtifactRatios)
        cleanToArtifactRatio = cleanToArtifactRatios(cleanToArtifactIdx);
        costWeight = computeClassWeights(Y_fs_train);
        costMatrix = [0 1; costWeight 0];
    
        for idx = 1:length(criteriaList)
            % Start timer
            startTime = datetime('now');

            criteria = criteriaList{idx}; 
            disp(['Evaluating using criterion: ', criteria, ', Ratio: ', num2str(cleanToArtifactRatio)])

            % Feature selection with SVM RBF kernel
            [selectedFeatures_FS, evalMetrics_train, svmModel] = featureSelectionWithUndersampling(X_fs_train, Y_fs_train, costMatrix, criteria, signal_ids_train, cleanToArtifactRatio);
            %[selectedFeatures_FS, evalMetrics_train , svmModel] = featureSelection(X_fs_train, Y_fs_train, costMatrix, criteria, trainPatientIds);

            % Convert to a model that can output probability scores
            svmProbModel = fitPosterior(svmModel);

            % Predict on validation dataset, help to select cost function
            [X_fs_val, Y_fs_val, signal_ids_val] = extractFeatureValues(XVal, YVal, artifactIdx, valPatientIds);
            X_fs_val = X_fs_val(:, selectedFeatures_FS);

            predictions = predict(svmModel, X_fs_val);
            evalMetrics_val = computeEvaluationMetrics(Y_fs_val, predictions);

            % To make predictions with soft labels (probabilities)
            [~, probScores] = predict(svmProbModel, X_fs_val);
            % Compute PR AUC using the function
            prAUC = computePRCurveAUC(Y_fs_val, probScores(:,2), 1);
            evalMetrics_val.prAUC = prAUC;

            % Predict on unseen dataset, for feature comparison with LSTM
            [X_fs_unseen, Y_fs_unseen, signal_ids_test] = extractFeatureValues(XTest, YTest, artifactIdx, testPatientIds);
            X_fs_unseen = X_fs_unseen(:, selectedFeatures_FS);

            predictions = predict(svmModel, X_fs_unseen);
            evalMetrics_unseen = computeEvaluationMetrics(Y_fs_unseen, predictions);

            % To make predictions with soft labels (probabilities)
            [~, probScores] = predict(svmProbModel, X_fs_unseen);
            % Compute PR AUC using the function
            prAUC = computePRCurveAUC(Y_fs_unseen, probScores(:,2), 1);
            evalMetrics_unseen.prAUC = prAUC;

            % End timer
            endTime = datetime('now');
            duration = endTime - startTime;

            % Save results to Excel File
            resultsTable = table(artifactIdx, ...
                strjoin(string(selectedFeatures_FS), ', '), strjoin(string(featNames(selectedFeatures_FS)), ', '), ...
                evalMetrics_train.accuracy, evalMetrics_train.sensitivity, evalMetrics_train.specificity, evalMetrics_train.precision, evalMetrics_train.f1, evalMetrics_val.youden, evalMetrics_train.prAUC, ...
                evalMetrics_val.accuracy, evalMetrics_val.sensitivity, evalMetrics_val.specificity, evalMetrics_val.precision, evalMetrics_val.f1, evalMetrics_val.youden, evalMetrics_val.prAUC, ...
                evalMetrics_unseen.accuracy, evalMetrics_unseen.sensitivity, evalMetrics_unseen.specificity, evalMetrics_unseen.precision, evalMetrics_unseen.f1, evalMetrics_unseen.youden, evalMetrics_unseen.prAUC, ...
                cleanToArtifactRatio, costWeight, string(criteria), ...
                startTime, duration, ...
                'VariableNames', {'artifactIdx', 'Selected_FS_Features', 'Selected_FS_Features_Names', ...
                                  'Accuracy_Train', 'Sensitivity_Train', 'Specificity_Train', 'Precision_Train', 'F1_Score_Train', 'Youden_Train', 'PR_AUC_Train', ...
                                  'Accuracy_Validation', 'Sensitivity_Validation', 'Specificity_Validation', 'Precision_Validation', 'F1_Score_Validation', 'Youden_Validation', 'PR_AUC_Validation', ...
                                  'Accuracy_Unseen', 'Sensitivity_Unseen', 'Specificity_Unseen', 'Precision_Unseen', 'F1_Score_Unseen', 'Youden_Unseen', 'PR_AUC_Unseen', ...
                                  'Clean_To_Artifact_Ratio', 'ClassWeight', 'Criterion', 'Start_Time', 'Duration'});
            saveResultsToExcel(excelFile, sheetName, resultsTable);
        end
    end
end