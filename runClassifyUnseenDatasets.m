clc; clear;

sigInspectAddpath;

%% Parameters
% Define paths
dataFolder = 'data/';
czskFolder = 'microrecordingCZSK-onlyMat';
% Metadata saved in mat file
matFile = 'microrecordingCZSK.mat';
matFilePath = fullfile(dataFolder, czskFolder, matFile);

loadedSignalsPath = fullfile(dataFolder, 'loadedSignalsCZSK.mat');
featureDataPath = fullfile(dataFolder, 'featureDataMergedCZSK.mat');

%% Load or create data and features suitable for model training
% LOAD OR CREATE SIGNALS AND ANNOTATIONS
if isfile(loadedSignalsPath)
    fprintf('Loaded signals file exists. Loading data...\n');
    load(loadedSignalsPath, 'loadedSignals');
else
    fprintf('Loaded signals file not found. Running `loadSignalsFromMat`...\n');
    loadedSignals = loadSignalsFromMat(matFilePath, true, loadedSignalsPath);
end
[signalData, annotationsData, signalIds, sampFrequencies] = extractSignalData(loadedSignals, true);

%% Delete Kosice patients
% Get patient IDs from signalIds
[patientIds, ~] = getPatientIds(signalIds);
validIndices = cellfun(@(x) ~startsWith(x, 'Kos'), patientIds);

signalData = signalData(validIndices);
annotationsData = annotationsData(validIndices);
signalIds = signalIds(validIndices);
sampFrequencies = sampFrequencies(validIndices);

%%
%LOAD OR CREATE FEATURE DATA (X and Y)
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
    [signalCellData, annotationsData, signalIds, sampFrequencies] = extractSignalData(loadedSignals);

    % Define feature computation parameters
    featNames = {'ksnorm', 'powDiff', 'maxNormPSD', 'stdNormPSD', ...
                 'maxAbsDiffPSD', 'psdF100', 'psdBase', 'psdFreq', 'energyEntrophy5', 'peakFreq', 'energyRatio'};
    windowLength = 1; % seconds

    % Compute features
    if iscell(sampFrequencies)
        samplFreqVec = cellfun(@(x) double(x(1)), sampFrequencies, 'UniformOutput', true);
    else
        samplFreqVec = 24000;
    end
    [X, Y, featNames] = computeFeaturesForLSTM(signalData, annotationsData, windowLength, samplFreqVec, featNames);

    % Save feature data
    featureSet = struct('X', {X}, ...
                     'Y', {Y}, ...
                     'featNames', {featNames}, ...
                     'signalIds', {signalIds});
    save(featureDataPath, 'featureSet', '-v7.3');
    fprintf('Features data saved to %s\n', featureDataPath);
end


%% Remove NANs
[~, nUniquePatients] = getPatientIds(signalIds);
fprintf('Number of patients before filtering: %d\n', nUniquePatients);
% Nan values can be removed based on raw signal data or feature data
validIdx = findValidIndices(signalData);
signalData = signalData(validIdx);
annotationsData = annotationsData(validIdx);
X = X(validIdx);
Y = Y(validIdx);
signalIds = signalIds(validIdx);
[~, nUniquePatients] = getPatientIds(signalIds);
fprintf('Number of patients after filtering: %d\n', nUniquePatients);

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

[~, nUniquePatients] = getPatientIds(signalIdsFiltered);
fprintf('Number of patients after filtering of unwanted artifact types: %d\n', nUniquePatients);


%% Removes windows from Y to match the number of windows in X
Yfiltered = matchWindowsXY(Xfiltered, Yfiltered);


%% Visualize artifact weights
counts = visualizeArtifactWeights(Xfiltered, Yfiltered, signalIdsFiltered,true)

%% Split data by centers
[patientIds, ~] = getPatientIds(signalIdsFiltered);
BrnIndices = cellfun(@(x) startsWith(x, 'Brn'), patientIds);
OloIndices = cellfun(@(x) startsWith(x, 'Olo'), patientIds);
BraIndices = cellfun(@(x) startsWith(x, 'Bra'), patientIds);


%% Load net and predict unseen test sets
 % Decide which artifact type to process - POW (2), BASE (3), FREQ (4)
SVMFile = fullfile(dataFolder, 'svm_results_FREQ.mat');
loaded = load(SVMFile, 'results_svm');
results_svm = loaded.results_svm;

% LSTMFile = fullfile(dataFolder, 'lstm_results_POW.mat');
% loaded = load(LSTMFile, 'results_lstm');
% results_lstm = loaded.results_lstm;

artifactIdx = 4; 
selectedFeatNames = {'ksnorm', 'maxNormPSD', 'stdNormPSD', 'maxAbsDiffPSD', 'psdBase', 'psdFreq'};
% 4 - {'ksnorm', 'maxNormPSD', 'stdNormPSD', 'maxAbsDiffPSD', 'psdBase', 'psdFreq'};
% 3 - {'stdNormPSD', 'psdF100', 'psdBase'};
% 2 - {'powDiff', 'maxNormPSD', 'maxAbsDiffPSD', 'energyEntrophy5', 'peakFreq', 'energyRatio'};

featuresIdx = cellfun(@(id) find(strcmp(id, featNames)), selectedFeatNames);
Xfinal = cellfun(@(x) x(featuresIdx, :), Xfiltered, 'UniformOutput', false);
Yfinal = Yfiltered; %cellfun(@(y) categorical(double(y(artifactIdx, :))), Yfiltered, 'UniformOutput', false);

%% Predict SVM
% BRNO
[X_fs_Brn, Y_fs_Brn, signal_ids_Brn] = extractFeatureValues(Xfinal(BrnIndices), Yfinal(BrnIndices), artifactIdx, signalIdsFiltered(BrnIndices));

predictions = predict(results_svm.svmModel, X_fs_Brn);
evalMetrics_Brn = computeEvaluationMetrics(Y_fs_Brn, predictions);
% To make predictions with soft labels (probabilities)
[~, predictedProbsBrn] = predict(results_svm.svmProbModel, X_fs_Brn);
% Compute PR AUC using the function
prAUC = computePRCurveAUC(Y_fs_Brn, predictedProbsBrn(:,2), 1);
evalMetrics_Brn.prAUC = prAUC;
% Compute ROC AUC
rocAUC = computeROCAUC(Y_fs_Brn, predictedProbsBrn(:,2), 1);
evalMetrics_Brn.rocAUC = rocAUC;

%disp(evalMetrics_Brn)

% OLOMOUC
[X_fs_Olo, Y_fs_Olo, signal_ids_Olo] = extractFeatureValues(Xfinal(OloIndices), Yfinal(OloIndices), artifactIdx, signalIdsFiltered(OloIndices));

predictions = predict(results_svm.svmModel, X_fs_Olo);
evalMetrics_Olo = computeEvaluationMetrics(Y_fs_Olo, predictions);
% To make predictions with soft labels (probabilities)
[~, predictedProbsOlo] = predict(results_svm.svmProbModel, X_fs_Olo);
% Compute PR AUC using the function
prAUC = computePRCurveAUC(Y_fs_Olo, predictedProbsOlo(:,2), 1);
evalMetrics_Olo.prAUC = prAUC;
% Compute ROC AUC
rocAUC = computeROCAUC(Y_fs_Olo, predictedProbsOlo(:,2), 1);
evalMetrics_Olo.rocAUC = rocAUC;

%disp(evalMetrics_Olo)

% BRATISLAVA
[X_fs_Bra, Y_fs_Bra, signal_ids_Bra] = extractFeatureValues(Xfinal(BraIndices), Yfinal(BraIndices), artifactIdx, signalIdsFiltered(BraIndices));

predictions = predict(results_svm.svmModel, X_fs_Bra);
evalMetrics_Bra = computeEvaluationMetrics(Y_fs_Bra, predictions);
% To make predictions with soft labels (probabilities)
[~, predictedProbsBra] = predict(results_svm.svmProbModel, X_fs_Bra);
% Compute PR AUC using the function
prAUC = computePRCurveAUC(Y_fs_Bra, predictedProbsBra(:,2), 1);
evalMetrics_Bra.prAUC = prAUC;
% Compute ROC AUC
rocAUC = computeROCAUC(Y_fs_Bra, predictedProbsBra(:,2), 1);
evalMetrics_Bra.rocAUC = rocAUC;

%disp(evalMetrics_Bra)
%% Predict LSTM
% miniBatchSize = 16;
% predictedProbsBrn = predict(results_lstm.net, Xfinal(BrnIndices), 'MiniBatchSize', miniBatchSize);
% predictedProbsOlo = predict(results_lstm.net, Xfinal(OloIndices), 'MiniBatchSize', miniBatchSize);
% predictedProbsBra = predict(results_lstm.net, Xfinal(BraIndices), 'MiniBatchSize', miniBatchSize);
%% Evaluate unseen test sets, only works for LSTM
mode = 'binary';
evalMetricsVal = evaluateModel(results_svm.predictedProbsVal, results_svm.YVal, mode, artifactIdx);

evalMetricsBrn = evaluateModel(predictedProbsBrn,  Yfinal(BrnIndices), mode, artifactIdx, evalMetricsVal.optimalThreshold);
evalMetricsOlo = evaluateModel(predictedProbsOlo, Yfinal(OloIndices), mode, artifactIdx, evalMetricsVal.optimalThreshold);
evalMetricsBra = evaluateModel(predictedProbsBra, Yfinal(BraIndices), mode, artifactIdx, evalMetricsVal.optimalThreshold);
%% Display ROC or PR curves for multiple datasets - unseen
positiveClassIdx = 2;

% For LSTM
% predictedProbsPraPos = cellfun(@(x) x(positiveClassIdx, :)', results_svm.predictedProbsTest, 'UniformOutput', false);
% predictedProbsBrnPos = cellfun(@(x) x(positiveClassIdx, :)', predictedProbsBrn, 'UniformOutput', false);
% predictedProbsOloPos = cellfun(@(x) x(positiveClassIdx, :)', predictedProbsOlo, 'UniformOutput', false);
% predictedProbsBraPos = cellfun(@(x) x(positiveClassIdx, :)', predictedProbsBra, 'UniformOutput', false);

% For SVM
predictedProbsPraPos = results_svm.predictedProbsTest(:,positiveClassIdx);
predictedProbsBrnPos = predictedProbsBrn(:,positiveClassIdx);
predictedProbsOloPos = predictedProbsOlo(:,positiveClassIdx);
predictedProbsBraPos = predictedProbsBra(:,positiveClassIdx);

predictedProbsAll = {predictedProbsPraPos,predictedProbsBrnPos, predictedProbsOloPos, predictedProbsBraPos}; 
labelsAll = {results_svm.YTest, Y_fs_Brn, Y_fs_Olo, Y_fs_Bra}; 
namesAll = {'Praha', 'Brno', 'Olomouc', 'Bratislava'};  

plotClassificationCurves(predictedProbsAll, labelsAll, 'roc', ...
   'DatasetNames', namesAll); %'Threshold', evalMetricsVal.optimalThreshold, ...

