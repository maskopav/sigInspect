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
%% Visualize artifact distribution
counts = visualizeArtifactWeights(Xfiltered, Yfiltered, signalIdsFiltered)

%% Artifacts undersampling

cfg(1).artifactIdx = 2; cfg(1).nToRemove = 500; cfg(1).patientId = 'sig_2';
cfg(2).artifactIdx = 3; cfg(2).nToRemove = 553; cfg(2).patientId = 'sig_17';
cfg(3).artifactIdx = 4; cfg(3).nToRemove = 394; cfg(3).patientId = 'sig_2';

[Xundersampled, Yundersampled, signalIdsUndersampled] = undersampleArtifactsMulti(Xfiltered, Yfiltered, signalIdsFiltered, cfg);

%%
counts = visualizeArtifactWeights(Xundersampled, Yundersampled, signalIdsUndersampled)
%%  Save undersmapled feature data
featureDataPath = fullfile(dataFolder, 'featureSetUndersampled.mat');
featureSetUndersampled = struct('X', {Xundersampled}, ...
                 'Y', {Yundersampled}, ...
                 'featNames', {featNames}, ...
                 'signalIds', {signalIdsUndersampled});
save(featureDataPath, 'featureSetUndersampled', '-v7.3');
fprintf('Features data saved to %s\n', featureDataPath);

%% Visualize sets after undersampling
% load(fullfile(dataFolder, 'featureSetUndersampled.mat'), 'featureSetUndersampled');
% 
% X = featureSetUndersampled.X;        % Cell array
% Y = featureSetUndersampled.Y;        % Cell array
% signalIds = featureSetUndersampled.signalIds; % Vector
% featNames = featureSetUndersampled.featNames; % Cell array

load(fullfile(dataFolder, 'featureDataMerged.mat'), 'featureSet');

X = featureSet.X;        % Cell array
Y = featureSet.Y;        % Cell array
signalIds = featureSet.signalIds; % Vector
featNames = featureSet.featNames; % Cell array
% Remove NANs 
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
% Convert annotations from numeric format to binary format
% multiclass or binary class option -> choose correct mode

mode = 'multi'; % or 'binary'
% Number of artifact types for type mode only
maxN = 6;

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


%%
% Data split for model training
ratios = struct('train', 0.65, 'val', 0.2, 'test', 0.15);
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIds, ratios);
[trainPatientIds, trainUniquePatients] = getPatientIds(signalIds(trainIdx));
[valPatientIds, valUniquePatients] = getPatientIds(signalIds(valIdx));
[testPatientIds, testUniquePatients] = getPatientIds(signalIds(testIdx));

% Display results
fprintf('Number of training samples: %d, number of unique patients: %d\n', numel(trainIdx), trainUniquePatients);
fprintf('Number of validation samples: %d, number of unique patients: %d\n', numel(valIdx), valUniquePatients);
fprintf('Number of test samples: %d, number of unique patients: %d\n', numel(testIdx), testUniquePatients);

% Final variables for the model
Xfinal = Xfiltered;
Yfinal = Yfiltered;
%Yfinal = cellfun(@(y) categorical(double(y(:)')), Yfinal, 'UniformOutput', false);
signalIdsFinal = signalIdsFiltered;

% Access the splits
XTrain = Xfinal(trainIdx, :);
YTrain = Yfinal(trainIdx, :);
XVal = Xfinal(valIdx, :);
YVal = Yfinal(valIdx, :);
XTest = Xfinal(testIdx, :);
YTest = Yfinal(testIdx, :);
%%

% Final variables for the model
Xfinal = X;
Yfinal = Y;
%Yfinal = cellfun(@(y) categorical(double(y(:)')), Yfinal, 'UniformOutput', false);
signalIdsFinal = signalIds;

% Create patientDatasetMap to know which patient belongs to which set
uniquePatientIds = unique(getPatientIds(signalIdsFinal));
patientDatasetMap = struct();

for i = 1:numel(uniquePatientIds)
    patientId = uniquePatientIds{i};
    if any(strcmp(trainPatientIds, patientId))
        patientDatasetMap.(patientId) = 'TRAIN';
    elseif any(strcmp(valPatientIds, patientId))
        patientDatasetMap.(patientId) = 'VALIDATION';
    elseif any(strcmp(testPatientIds, patientId))
        patientDatasetMap.(patientId) = 'TEST';
    else
        patientDatasetMap.(patientId) = 'UNKNOWN';
        warning('Patient ID "%s" not found in any split.', patientId);
    end
end

counts = visualizeArtifactWeights(Xfinal, Yfinal, signalIdsFinal, true, patientDatasetMap)

%% Check if the 7th row in all X cells is zeros everywhere (maxCorr feature)
rowToDelete = 7;
isRowZero = cellfun(@(m) all(m(rowToDelete, :) == 0), X);

if all(isRowZero)
    fprintf('The 7th row is zero everywhere. Deleting row and corresponding feature name...\n');
    
    % Remove the 7th row from all matrices in X
    X = cellfun(@(m) m([1:rowToDelete-1, rowToDelete+1:end], :), X, 'UniformOutput', false);
    
    % Remove corresponding feature name
    featNames(rowToDelete) = [];
else
    warning('The 7th row is NOT zero everywhere. Aborting deletion.');
end


%%
% Find indices where signalIds start with 'sig_17' or 'sig_2'
indicesToDelete = find( ...
    startsWith(signalIds, 'sig_17') | startsWith(signalIds, 'sig_2') ...
);

% Delete those indices from all arrays
X(indicesToDelete) = [];
Y(indicesToDelete) = [];
signalIds(indicesToDelete) = [];

emptyIdx = cellfun(@isempty, X);
X = X(~emptyIdx);
Y = Y(~emptyIdx);
signalIds = signalIds(~emptyIdx);
