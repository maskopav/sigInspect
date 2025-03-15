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

%% AUC (perfcurve) for feature evaluation
% Every feature own plot..
numFeatures = size(Xfinal{1}, 1); % Number of features
artifactIdx = 5;

figure;
for featIdx = 1:numFeatures
    % Collect feature values and corresponding Y labels
    featureValues = [];
    labels = [];

    for i = 1:length(Xfinal)
        % Extract feature values and labels
        featVals = Xfinal{i}(featIdx, :);
        labelVals = Yfinal{i}(artifactIdx, :);  % Artifact labels (0 or 1)
        cleanMask = Yfinal{i}(1, :) == 1;       % Only evaluate where cleanMask is true
        
        % Apply the condition: keep 0 only if cleanMask is true, otherwise NaN
        labelVals(~cleanMask & labelVals == 0) = NaN;
        
        % Store values
        featureValues = [featureValues, featVals]; 
        labels = [labels, labelVals]; %36200 vs 28773
    end

    % Separate data for Y=0 and Y=1
    feature_0 = log1p(featureValues(labels == 0));
    feature_1 = log1p(featureValues(labels == 1));

    % Estimate PDFs using kernel density estimation (ksdensity)
    [pdf_0, x_0] = ksdensity(feature_0);
    [pdf_1, x_1] = ksdensity(feature_1);

    % Compute AUC
    [X_ROC, Y_ROC, ~, AUC] = perfcurve(labels, featureValues, 1);

    % Plot histograms
    subplot(ceil(sqrt(numFeatures)), ceil(sqrt(numFeatures)), featIdx);
    hold on;
    histogram(feature_0, 'Normalization', 'pdf', 'FaceColor', 'b', 'FaceAlpha', 0.5);
    histogram(feature_1, 'Normalization', 'pdf', 'FaceColor', 'r', 'FaceAlpha', 0.5);

    plot(x_0, pdf_0, '-', 'LineWidth', 1);
    plot(x_1, pdf_1, '--', 'LineWidth', 1);
    hold off;

    % Title with AUC
    title(sprintf('Feature %s - AUC: %.2f', featNames{featIdx}, AUC));
    xlabel('Feature Value');
    ylabel('Probability');
    legend({'Y=0', 'Y=1'});
end

sgtitle('Histograms of Features with AUC'); % Add a main title
 
%% Plot only n top features based on AUC
artifactIdx = 5;
numTopFeatures = 7; % Set the number of top features to plot

[AUC_values, selectedFeatures_AUC, allFeatureValues, labels] = computeAUC(Xfinal, Yfinal, artifactIdx, numTopFeatures, featNames);

%%
%%% Feature selection with SVM RBF kernel
% Remove NaN values
validIdx = ~isnan(labels);
X_fs = allFeatureValues(:,validIdx)';  % Features (rows: samples, cols: features)
Y_fs = categorical(labels(validIdx)'); % Labels (0 = clean, 1 = artifact)


costMatrix = [0 1/(sum(double(string(Y_fs)))/length(Y_fs)); 1 0];

[selectedFeatures_FS, accuracy, sensitivity, specificity, precision, f1_score] = featureSelection(X_fs, Y_fs, costMatrix);


% Save Results to Excel File
% Define the Excel file and sheet name
excelFile = 'Feature_selection_results.xlsx';
sheetName = 'FS';

% Create a table with all results
resultsTable = table(artifactIdx, ...
    strjoin(string(selectedFeatures_AUC), ', '), strjoin(string(featNames(selectedFeatures_AUC)), ', '), ...
    strjoin(string(selectedFeatures_FS), ', '), strjoin(string(featNames(selectedFeatures_FS)), ', '), ...
    accuracy, sensitivity, specificity, precision, f1_score, strjoin(string(costMatrix), ', '), ...
    'VariableNames', {'artifactIdx', 'Selected_AUC_Features', 'Selected_AUC_Features_Names', ...
                      'Selected_FS_Features', 'Selected_FS_Features_Names', ...
                      'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'Cost_Matrix'});

% Check if the file already exists
if isfile(excelFile)
    % Read existing data
    try
        existingData = readtable(excelFile, 'Sheet', sheetName);
    catch
        existingData = table(); % If sheet doesn't exist, create an empty table
    end
    % Append the new results to the existing data
    updatedTable = [existingData; resultsTable]; 
else
    % If the file doesn't exist, just use the new results
    updatedTable = resultsTable;
end

% Write the updated table to Excel
writetable(updatedTable, excelFile, 'Sheet', sheetName);


% Define the dynamically named sheet for the plot
sheetName = sprintf('Feature_PDFs_%d', artifactIdx); 
plotFileName = sprintf('Feature_PDFs_%d.png', artifactIdx); % Save the plot as PNG

%%% PLOT THE PDFs OF SELECTED TOP FEATURES


figure; hold on;
colors = lines(numTopFeatures); % Generate distinct colors for each feature

legendEntries = cell(2*numTopFeatures, 1); % Preallocate legend entries

for i = 1:numTopFeatures
    featIdx = selectedFeatures_AUC(i);
    topFeatureValues = allFeatureValues(featIdx,:);
    disp(featNames{featIdx})

    % Separate data for Y=0 and Y=1
    feature_0 = topFeatureValues(labels == 0);
    feature_1 = topFeatureValues(labels == 1);

    % Check normality using skewness
    if abs(skewness(topFeatureValues)) > 2
        disp(skewness(topFeatureValues))
        disp('Log transformation')
        % Handle negative values before log transformation
        min_value = min([feature_0, feature_1]);
        if min_value <= 0
            feature_0 = log1p(feature_0 - min_value + 0.01);
            feature_1 = log1p(feature_1 - min_value + 0.01);
        else 
            feature_0 = log1p(feature_0);
            feature_1 = log1p(feature_1);
        end
    end

    if isempty(feature_0) || isempty(feature_1)
        warning('Skipping feature %d: One of the label groups is empty.', featIdx);
        continue;
    end

    % Estimate PDFs using kernel density estimation (ksdensity)
    [pdf_0, x_0] = ksdensity(feature_0);
    [pdf_1, x_1] = ksdensity(feature_1);

    plot(x_0, pdf_0, '-', 'Color', colors(i, :), 'LineWidth', 2);
    plot(x_1, pdf_1, '--', 'Color', colors(i, :), 'LineWidth', 2);

    legendEntries{(2*i-1)} = sprintf('%s 0 - AUC: %.2f', featNames{featIdx}, AUC_values(featIdx));
    legendEntries{(2*i)} = sprintf('%s 1 - AUC: %.2f', featNames{featIdx}, AUC_values(featIdx));
end

legend(legendEntries, 'Location', 'Best');
xlabel('Feature value');
ylabel('Probability density');
title(sprintf('PDFs of %d top features by AUC, Artifact %d', numTopFeatures, artifactIdx));
hold off;

% Save the figure
saveas(gcf, plotFileName);
%9.14
% 14:51

% artif=5 -> 4 10 20
%% Signal indicies by types of artifact
cleanIndicies = find(cellfun(@(y) any(y(1, :)), Yfinal));
powerArtifIndices = find(cellfun(@(y) any(y(2, :)), Yfinal));
baseArtifIndices = find(cellfun(@(y) any(y(3, :)), Yfinal));
freqArtifIndices = find(cellfun(@(y) any(y(4, :)), Yfinal));
irritArtifIndices = find(cellfun(@(y) any(y(5, :)), Yfinal));

%% Visualize the desired signal with features 
cellIdx = cleanIndicies(7);

samplingFrequency = 24000;
windowLength = 1;
windowLengthSamples = samplingFrequency * windowLength;
sampleSignal = signalData{cellIdx};
sampleFeatures = Xfinal{cellIdx};

selectedFeaturesIdx = (14:19);
sampleFeatures = sampleFeatures(selectedFeaturesIdx, :);
sampleFeatNames = featNames(selectedFeaturesIdx);

% Normalize features between 0 and 1
sampleNormFeatures = normalizeFeatures(sampleFeatures);

visualizeSignalWithFeatures(sampleSignal, samplingFrequency, sampleNormFeatures, sampleFeatNames, windowLengthSamples, true);

%% Compute and visualize new features
% cleanIndicies, irritArtifIndices 9 
cellIdx = powerArtifIndices(19);

samplingFrequency = 24000;
windowLength = 1;
windowLengthSamples = samplingFrequency * windowLength;
sampleSignal = signalData{cellIdx};

sampleWindows = divideIntoWindows(sampleSignal, windowLength, samplingFrequency);
numWindows = size(sampleWindows, 3);
sampleFeatNames = {'energyEntrophy5', 'energyEntrophy15','peakToRMS',...
    'HjorthMobility', 'HjorthComplexity', 'numPeaks', 'meanPeakHeight','peakFreq',...
    'peakRMSRatio', 'avgPeakWidth', 'energyRatio', 'sparseness', 'irregularity', 'zeroUpCrossingPeriod','sigLen'};
sampleFeatures = zeros(numel(sampleFeatNames), numWindows); % Preallocate feature matrix

% Compute features for each window in parallel
for j = 1:numWindows
    window = sampleWindows(:,:,j); % Extract each normalized window

    % Features computation
    sampleFeatures(1, j) = compShannonEntropy(window, 5);
    sampleFeatures(2, j) = compShannonEntropy(window, 15);
    sampleFeatures(3, j) = compPeakToRMS(window);
    sampleFeatures(4, j) = compHjorthMobility(window);
    sampleFeatures(5, j) = compHjorthComplexity(window);
    smoothWindowDuration = 0.005;
    [numPeaks, meanPeakHeight, peakFreq, peakRMSRatio, avgPeakWidth] = compPowerPeakFeatures(window, samplingFrequency, smoothWindowDuration, false);
    sampleFeatures(6, j) = numPeaks;
    sampleFeatures(7, j) = meanPeakHeight;
    sampleFeatures(8, j) = peakFreq;
    sampleFeatures(9, j) = peakRMSRatio;
    sampleFeatures(10, j) = avgPeakWidth;
    sampleFeatures(11, j) = computeEnergyRatio(window, samplingFrequency);
    sampleFeatures(12, j) = compSparseness(window);
    sampleFeatures(13, j) = compIrregularityFactor(window);
    sampleFeatures(14, j) = compZeroUpCrossingPeriod(window, samplingFrequency, smoothWindowDuration);
    sampleFeatures(15, j) = compSignalLength(window);

end

% Normalize features between 0 and 1
sampleNormFeatures = normalizeFeatures(sampleFeatures);

selectedFeaturesIdx = (11:15);
sampleNormFeatures = sampleNormFeatures(selectedFeaturesIdx, :);
sampleFeatNames = sampleFeatNames(selectedFeaturesIdx);

visualizeSignalWithFeatures(sampleSignal, samplingFrequency, sampleNormFeatures, sampleFeatNames, windowLengthSamples, false);

%% Compute new features and merge them with existing ones
% Extract signal and annotation data
[signalCellData, annotationsData, signalIds] = extractSignalData(loadedSignals);

% Define feature computation parameters
featNames = {'energyEntrophy5', 'energyEntrophy15','peakToRMS',...
'HjorthMobility', 'HjorthComplexity', 'numPeaks', 'meanPeakHeight','peakFreq',...
'peakRMSRatio', 'avgPeakWidth', 'energyRatio', 'sparseness', 'irregularity', 'zeroUpCrossingPeriod','sigLen'};
samplingFreq = 24000; % Hz
windowLength = 1; % seconds

disp('Computation of new features has started')

% Compute features
[X, Y, featNames] = computeFeaturesForLSTM(signalCellData, annotationsData, windowLength, samplingFreq, featNames);

featureDataPath = fullfile(dataFolder, 'featureDataNew.mat');
% Save feature data
featureSetNew = struct('X', {X}, ...
                 'Y', {Y}, ...
                 'featNames', {featNames}, ...
                 'signalIds', {signalIds});
save(featureDataPath, 'featureSetNew', '-v7.3');
fprintf('Features data saved to %s\n', featureDataPath);

%% LOAD new features and merge them 
featureDataPath = fullfile(dataFolder, 'featureDataNew.mat');
load(featureDataPath, 'featureSetNew');

XNew = featureSetNew.X;        % Cell array
YNew = featureSetNew.Y;        % Cell array
signalIdsNew = featureSetNew.signalIds; % Vector
featNamesNew = featureSetNew.featNames; % Cell array

%%
mergedX = X;  % Initialize with old data
mergedSignalIds = signalIds;  % Initialize with old signal IDs

% Loop through new signal IDs
for i = 1:length(signalIdsNew)
    id = signalIdsNew{i};  % Get the current signal ID
    
    % Find index of matching signalId in old data
    idxOld = find(cellfun(@(x) isequal(x, id), signalIds), 1);

    if ~isempty(idxOld)  % If the ID exists in old data
        mergedX{idxOld} = [X{idxOld}; XNew{i}];  % Merge features (concatenate rows)
    else
        % If new ID is not found in old set, append it
        mergedX{end+1} = XNew{i};
        mergedSignalIds(end+1) = id;
    end
end

mergedFeatNames = [featNames, featNamesNew];
fprintf('Unique IDs before merging: %d\n', numel(signalIds));
fprintf('Unique IDs in new dataset: %d\n', numel(signalIdsNew));
fprintf('Unique IDs after merging: %d\n', numel(mergedSignalIds));

featureDataPath = fullfile(dataFolder, 'featureDataMerged.mat');
% Save feature data
featureSet = struct('X', {mergedX}, ...
                 'Y', {Y}, ...
                 'featNames', {mergedFeatNames}, ...
                 'signalIds', {mergedSignalIds});
save(featureDataPath, 'featureSet', '-v7.3');
fprintf('Features data saved to %s\n', featureDataPath);





