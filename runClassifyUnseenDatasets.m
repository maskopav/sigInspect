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
[signalData, annotationsData, signalIds, sampFrequencies] = extractSignalData(loadedSignals);
%%

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
    % % Extract signal and annotation data
    % [signalCellData, annotationsData, signalIds, sampFrequencies] = extractSignalData(loadedSignals);

    % Define feature computation parameters
    featNames = {'pow', 'sigP90', 'sigP95', 'sigP99', 'ksnorm', 'powDiff', 'maxCorr', ...
                 'maxNormPSD', 'stdNormPSD', 'psdP75', 'psdP90', 'psdP95', 'psdP99', ...
                 'maxAbsDiffPSD', 'psdF100', 'psdBase', 'psdPow', 'psdFreq', 'psdMaxStep'};
    windowLength = 1; % seconds

    % Compute features
    samplFreqVec = cellfun(@(x) double(x(1)), sampFrequencies, 'UniformOutput', true);
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

%% Visualize 

counts = visualizeArtifactWeights(Xfiltered, Yfiltered, signalIdsFiltered)