clc; clear;

sigInspectAddpath;

%% Parameters
% Define paths
dataFolder = 'data/';
csvFile = '_metadataMER2020.csv';
signalsFolder = 'signals/';

loadedSignalsPath = fullfile(dataFolder, 'loadedSignals.mat');
featureDataPath = fullfile(dataFolder, 'featureData.mat');

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

%% AUC (perfcurve) for feature evaluation


%% FEATURE FUNCTIONS
function f = compSignalLength(segment)
    f = sum(abs(diff(segment,1,2)), 2); 
end

function entropyVal = compShannonEntropy(segment, numBins)
    % Random signals large entropy
    % H=−∑pi*log2pi
    if nargin < 2
        numBins = 5;
    end
    energy = segment .^ 2;
    probDist = histcounts(energy, numBins, 'Normalization', 'probability');
    probDist(probDist == 0) = []; % Remove zeros to avoid log(0)
    entropyVal = -sum(probDist .* log2(probDist));
end

function f = compPeakToRMS(segment)
    % Peak to peak vs RMS 
    % high for peaks
    peakToPeak = max(segment,[],2) - min(segment,[],2);
    rmsValue = sqrt(mean(segment.^2,2));
    f = peakToPeak ./ rmsValue;
end

function [numPeaks, meanPeakHeight, peakFreq, peakRMSRatio, avgPeakWidth] = compPowerPeakFeatures(signal, fs, smoothWindowDuration, plotOption)
    powerSignal = signal.^ 2;
    % Gaussian smoothing preserves peak structure better than a MA
    windowSize = round(smoothWindowDuration * fs);
    gaussKernel = gausswin(windowSize) / sum(gausswin(windowSize)); 
    smoothSignal = filtfilt(gaussKernel, 1, powerSignal);

    % Adaptive thresholding based on signal variation
    baselineNoise = median(abs(smoothSignal - median(smoothSignal))); % Robust estimate
    minProm = 1.5 * baselineNoise;  % Adaptive threshold
    minHeight = median(smoothSignal) + 8 * baselineNoise;
    minDistance = round(fs * 0.004); %s
    [pks, locs, width, ~] = findpeaks(smoothSignal, ...
        'MinPeakProminence', minProm, ...
        'MinPeakHeight', minHeight, ...
        'MinPeakDistance', minDistance);
    % Handle case where no peaks are found
    if isempty(pks)
        pks = NaN;
        locs = NaN;
        numPeaks = 0;
        meanPeakHeight = 0;
        avgPeakWidth = 0;
        peakFreq = 0;
    else
        % Features
        numPeaks = numel(pks);
        meanPeakHeight = mean(pks, 'omitnan'); 
        avgPeakWidth = mean(width, 'omitnan');
        if numPeaks > 1
            peakIntervals = diff(locs) / fs; 
            peakFreq = 1 / mean(peakIntervals);
        else
            peakFreq = 0;
        end
    end
    peakRMSRatio = meanPeakHeight / rms(smoothSignal);

    if plotOption
        timeVector = (0:length(signal)-1) / fs; 
        figure;
        plot(timeVector, signal, 'b'); 
        grid on; hold on;
        plot(timeVector, smoothSignal, '--r', 'LineWidth', 1); 
        if numPeaks > 0
            scatter(timeVector(locs), pks, 'go', 'MarkerFaceColor', 'g');
        end
        xlabel('Time (s)');
        ylabel('Amplitude');
        title(sprintf(['N Peaks: %d | Mean height: %.2f | Freq: %.2f Hz | Peak RMS Ratio: %.2f\n' ...
            'Mean width: %.2f'], ...
            numPeaks, meanPeakHeight, peakFreq, peakRMSRatio, avgPeakWidth));
        legend({'Original signal', 'Smoothed power signal', 'Peaks'});
        grid on; hold off;
    end
end

function mobility = compHjorthMobility(segment)
    % Frequency content and smoothness of the signal
    % High for random noise
    deriv = diff(segment);
    mobility = sqrt(var(deriv) / (var(segment) + eps));
end

function complexity = compHjorthComplexity(segment)
    % How much the frequency content changes over time
    % High for irregular signal
    deriv1 = diff(segment);
    deriv2 = diff(deriv1);
    mobility1 = compHjorthMobility(deriv1);
    mobility2 = compHjorthMobility(deriv2);
    complexity = mobility2 / (mobility1 + eps);
end

function sparseness = compSparseness(segment)
    % Measures how concentrated the energy is in few large values
    N = length(segment);
    sparseness = sqrt(sum(segment.^2) / N) / (sum(abs(segment)) / N);
end

function irregularity = compIrregularityFactor(segment)
    % Measures how different the signal is from white noise
    complexity_signal = compHjorthComplexity(segment);
    white_noise = randn(size(segment)); % Simulated white noise
    complexity_noise = compHjorthComplexity(white_noise);
    irregularity = complexity_signal / (complexity_noise + eps);
end

function zup = compZeroUpCrossingPeriod(segment, fs, smoothWindowDuration)
    % Computes the Zero Up-Crossing Period (ZUCP)
    % signal - input signal
    % fs - sampling frequency
    % smoothWindow - number of samples for smoothing (set to 1 for no smoothing)
    
    if smoothWindowDuration > 1
        smoothWindow = round(smoothWindowDuration * fs);
        b = ones(1, smoothWindow) / smoothWindow; % MA
        segment = filtfilt(b, 1, segment);
    end

    % Zero crossings - from negative to positive
    zeroCrossings = find(segment(1:end-1) < 0 & segment(2:end) >= 0);

    % Compute periods between consecutive zero-crossings
    if length(zeroCrossings) < 2
        zup = NaN;
        return;
    end
    periods = diff(zeroCrossings) / fs; % seconds
    zup = mean(periods);
end

function energyRatio = computeEnergyRatio(signal, fs, spikeBand)
    % Computes the energy ratio in a specified spike band relative to total power
    if nargin < 3
        spikeBand = [300 3000]; % Default spike band if not provided
    end

    % Compute power in the spike band
    spikePower = bandpower(signal, fs, spikeBand);  
    % Compute total power of the signal
    totalPower = bandpower(signal, fs, [0 fs/2]); 

    % Prevent division by zero
    if totalPower == 0
        energyRatio = 0;
    else
        energyRatio = spikePower / totalPower;
    end
end


