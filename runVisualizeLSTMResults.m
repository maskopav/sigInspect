clc; clear;
sigInspectAddpath;

%% Parameters
dataFolder = 'data/';

samplingFreq = 24000; % Hz
windowLength = 1; % sec
windowsPerSignal = 10; % number of windows per signal

%% Load Data - LSTM results, signals, features
% Decide which artfact type to process - POW (2), BASE (3), FREQ (4)
artifactIdx = 2;
resultsFile = fullfile(dataFolder, 'lstm_results_POW.mat');

loadedSignalsFile = fullfile(dataFolder, 'loadedSignals.mat');
featureDataPath = fullfile(dataFolder, 'featureSetUndersampled.mat');

results_lstm = loadLSTMResults(resultsFile);
[signalData, annotationsData, signalIdsOrig] = loadSignalData(loadedSignalsFile);
[X, Y, signalIds, featNames] = loadFeatures(featureDataPath);

%% Filter original signal data by signalIds from undersampled feature set, feature set was undersample, we need to unify that
% Match signalIds from undersampled feature set to original signalIds
[matchedTF, matchedIdx] = ismember(signalIds, signalIdsOrig);

% Only take those that matched
validIdx = matchedIdx(matchedTF);

% Filter signal data and annotations data
signalDataMatched = signalData(validIdx);
annotationsDataMatched = annotationsData(validIdx);
signalIdsMatched = signalIdsOrig(validIdx);

%% Visualizations 
% -> Evaluation results
mode = 'binary';
evalMetricsVal = evaluateModel(results_lstm.predictedProbsVal, results_lstm.YVal, mode, artifactIdx);
evalMetricsTrain = evaluateModel(results_lstm.predictedProbsTrain, results_lstm.YTrain, mode, artifactIdx, evalMetricsVal.optimalThreshold);
evalMetricsTest = evaluateModel(results_lstm.predictedProbsTest, results_lstm.YTest, mode, artifactIdx, evalMetricsVal.optimalThreshold);

%%
% -> Plot Soft Label Distribution (Unseen Test) - wont work if there are
% different number of windows across the signals
plotSoftLabelDistributions(results_lstm.predictedProbsTest, results_lstm.YTest);
%%
% -> Atlas of Signals Binned by Soft Labels (Test set)
% Data split for model training
ratios = struct('train', 0.65, 'val', 0.2, 'test', 0.15);
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIds, ratios);

visualizeSignalsBySoftLabelWithSpectrograms(results_lstm.predictedProbsTest, ...
    results_lstm.YTest, signalIdsMatched(testIdx), signalDataMatched(testIdx), samplingFreq, windowLength, windowsPerSignal);

%% Check the results
checkIdx = find(strcmp('sig_8t216p20p',signalIdsMatched(testIdx)));
cell2mat(results_lstm.predictedProbsTest(checkIdx))
results_lstm.YTest(checkIdx)

%%
% -> Display ROC or PR curves for multiple datasets
positiveClassIdx = 2;
predictedProbsTrain = cellfun(@(x) x(positiveClassIdx, :)', results_lstm.predictedProbsTrain, 'UniformOutput', false);
predictedProbsVal = cellfun(@(x) x(positiveClassIdx, :)', results_lstm.predictedProbsVal, 'UniformOutput', false);
predictedProbsTest = cellfun(@(x) x(positiveClassIdx, :)', results_lstm.predictedProbsTest, 'UniformOutput', false);

predictedProbsAll = {predictedProbsTrain, predictedProbsVal, predictedProbsTest};
labelsAll = {results_lstm.YTrain, results_lstm.YVal, results_lstm.YTest};
namesAll = {'Train', 'Validation', 'Unseen'};

mode = 'binary';
evalMetricsVal = evaluateModel(results_lstm.predictedProbsVal, results_lstm.YVal, mode, artifactIdx);
plotClassificationCurves(predictedProbsAll, labelsAll, 'roc', ...
   'Threshold', evalMetricsVal.optimalThreshold, ...
   'Colors',  {[0.8 0.2 0.3], [1.0 0.85 0.0], [0 0 1]}, ...
   'DatasetNames', namesAll);

%% Helper Functions

function results_lstm = loadLSTMResults(resultsFile)
    fprintf('Loading LSTM results from %s\n', resultsFile);
    loaded = load(resultsFile, 'results_lstm');
    results_lstm = loaded.results_lstm;
end

function [signalData, annotationsData, signalIdsOrig] = loadSignalData(loadedSignalsFile)
    fprintf('Loading signal data from %s\n', loadedSignalsFile);
    loaded = load(loadedSignalsFile, 'loadedSignals');
    [signalData, annotationsData, signalIdsOrig] = extractSignalData(loaded.loadedSignals);
end

function [X, Y, signalIds, featNames] = loadFeatures(featureDataPath)
    fprintf('Feature file exist. Loading data...\n');
    load(featureDataPath, 'featureSetUndersampled');
    X = featureSetUndersampled.X;        % Cell array
    Y = featureSetUndersampled.Y;        % Cell array
    signalIds = featureSetUndersampled.signalIds; % Cell array
    featNames = featureSetUndersampled.featNames; % Cell array
end

function visualizeSignalsBySoftLabelWithSpectrograms(predictedProbs, Y, signalIdsOrig, signalData, fs, winLenSec, windowsPerSignal)
    % This function visualizes signals from different soft label bins and their spectrograms
    % 
    % Inputs:
    %   predictedProbs - Cell array of predicted probabilities
    %   Y - Cell array of true labels
    %   signalIdsOrig - Original signal IDs
    %   signalData - Cell array of signal data
    %   fs - Sampling frequency (Hz)
    %   winLenSec - Window length in seconds
    %   windowsPerSignal - Number of windows per signal
    
    % Convert labels to numeric format
    Y = cellfun(@(y) double(string(y)), Y, 'UniformOutput', false);
    
    % Extract the labels, predicted probabilities, and signal IDs
    [trueLabels, predictedLabels, signalIdsLabels] = extractFeatureValues(Y, predictedProbs, 2, signalIdsOrig);
    disp(unique(trueLabels))
    
    % Define bin edges for soft label ranges
    binEdges = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    binLabels = {'0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'};
    
    % Assign windows to bins based on predicted probability
    binIdx = discretize(predictedLabels, binEdges);
    
    numBins = numel(binLabels);
    binnedWindows = cell(numBins, 1);
    
    % Collect windows for each bin
    for b = 1:numBins
        % Find windows in bin b with true label = 1 (artifact)
        winIdx = find(binIdx == b & trueLabels == 1);
        
        if ~isempty(winIdx)
            rng(11)
            % Select a random window from each bin
            randIdx = winIdx(randi(numel(winIdx)));
            %randIdx = winIdx(6); %FREQ 5
            disp(signalIdsLabels(randIdx))
            
            % Compute which signal and window within signal
            signalIdx = ceil(randIdx / windowsPerSignal);
            windowNum = mod(randIdx-1, windowsPerSignal) + 1
            signal = signalData{signalIdx};
            
            % Divide the signal into windows
            windows = divideIntoWindows(signal, winLenSec, fs);
            
            % Store the selected window
            binnedWindows{b} = windows(:,:,windowNum);
        else
            % If no windows in this bin, use an empty array
            binnedWindows{b} = [];
        end
    end
    
    % Create figure with 5 rows and 2 columns
    figure('Position', [100, 100, 1200, 800]);
    
    % For each bin, plot the signal and its spectrogram
    for b = 1:numBins
        window = binnedWindows{b};
        
        if ~isempty(window)
            % Calculate time vector for the window
            t = (0:length(window)-1) / fs;

            % Plot the signal in the first column
            subplot(numBins, 2, 2*b-1);
            plot(t, window, 'LineWidth', 1);
            title(sprintf('Soft label range %s', binLabels{b}));
            xlabel('Time (s)');
            ylabel('Amplitude');
            grid on;
            xlim([0 t(end)]);
            ylim([-300 300]); % Consistent y-axis limits
            
            % Plot the spectrogram in the second column
            subplot(numBins, 2, 2*b);
            computeAndPlotSpectrogram(window, fs, true);
            title(sprintf('Soft label range %s', binLabels{b}));
            ylabel('Freq (Hz)')
        else
            % If no windows in this bin, show empty plots
            subplot(numBins, 2, 2*b-1);
            text(0.5, 0.5, 'No signal available in this bin', 'HorizontalAlignment', 'center');
            axis off;
            
            subplot(numBins, 2, 2*b);
            text(0.5, 0.5, 'No spectrogram available in this bin', 'HorizontalAlignment', 'center');
            axis off;
        end
    end
    
    % Add overall title
    sgtitle('Typical signals in time domain and their spectrograms by soft label range');
end



% function visualizeSignalsBySoftLabel(predictedProbs, Y, signalIdsOrig, signalData, fs, winLenSec, windowsPerSignal)
%     Y = cellfun(@(y) double(string(y)), Y, 'UniformOutput', false);
%     [trueLabels, predictedLabels, signalIdsLabels] = extractFeatureValues(Y, predictedProbs, 2, signalIdsOrig);
% 
%     binEdges = [0, 0.2, 0.4, 0.6, 0.8];
%     binLabels = {'0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8'};
%     binIdx = discretize(predictedLabels, binEdges);
% 
%     numBins = numel(binLabels);
%     binnedWindows = cell(numBins, 1);
%     alphaVal = 0.3; % adjust transparency (0=transparent, 1=opaque)
% 
%     for b = 1:numBins
%         % Indices of windows in bin b with true label = 1 (artifact)
%         winIdx = find(binIdx == b & trueLabels == 1);
%         binnedWindows{b} = cell(numel(winIdx), 1);
%         for j = 1:numel(winIdx)
%             i = winIdx(j);
%             % Compute which signal and window within signal
%             signalIdx = ceil(i / windowsPerSignal);
%             windowNum = mod(i-1, windowsPerSignal) + 1;
%             signal = signalData{signalIdx};
%             windows = divideIntoWindows(signal, winLenSec, fs);
%             % Extract correct window
%             binnedWindows{b}{j} = windows(:,:,windowNum);
%         end
%     end
%     % Plotting
%     figure;
%     for b = 1:numBins
%         subplot(2,2,b); hold on;
%         title(sprintf('Soft Label Bin %s', binLabels{b}));
%         xlabel('Samples'); ylabel('Amplitude');
%         ylim([-300 300])
% 
%         numWindows = numel(binnedWindows{b});
%         if numWindows == 0
%             continue;
%         end
% 
%         % Create colormap (reuse if fewer than 7 windows)
%         cmap = lines(max(7, numWindows)); 
% 
%         for j = 1:min(2, numWindows)
%             window = binnedWindows{b}{j};
% 
%             % Get color for j-th plot (wrap around if needed)
%             baseColor = cmap(mod(j-1, size(cmap,1)) + 1, :);
%             lightColor = (1 - alphaVal) * [1 1 1] + alphaVal * baseColor;
% 
%             plot(window, 'Color', baseColor);
%         end
%         hold off;
%         xlim([0 length(window)])
%         ylim([-300 300])
%     end
% end
