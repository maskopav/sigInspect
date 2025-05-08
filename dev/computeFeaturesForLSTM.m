function [X, Y, featNames] = computeFeaturesForLSTM(signals, labels, windowLength, samplingFreq, featNames)
    % INPUTS:
    % signals: Cell array of signals, each signal is [channels x timePoints].
    % labels: Cell array of windows annotation for each signal (1 for artifact, 0 for non-artifact) [1 x windows].
    % windowLength: Length of each window (in seconds).
    % samplingFreq: Sampling frequency of the signals. If scalar, the same sampling frequency is used for all signals.
    %               If vector, it must have the same number of elements as 'signals',
    %               and each element corresponds to the sampling frequency of themcorresponding signal.
    % featNames: Names of features to compute for each window.
    %
    % OUTPUTS:
    % X: Cell array where each cell is a matrix [features x windows].
    % Y: Categorical labels for each window (binary).

    numSignals = numel(signals);
    X = cell(numSignals, 1);
    Y = cell(numSignals, 1);

    % Check if samplingFreq is a scalar or a vector
    if isscalar(samplingFreq)
        % If scalar, create a vector with the same value for all signals
        samplingFreqs = repmat(samplingFreq, 1, numSignals);
    elseif isvector(samplingFreq) && numel(samplingFreq) == numSignals
        % If vector and has the correct number of elements, use it directly
        samplingFreqs = samplingFreq;
    else
        error('samplingFreq must be either a scalar or a vector with the same number of elements as signals.');
    end

    % Progress tracking
    fprintf('Computing features for %d signals...\n', numSignals);
    progressMarks = round([0.01, 0.1, 0.25, 0.50, 0.75, 1] * numSignals);

    % Omit Kosice center - first patient is on index = 1930
    % find(strcmp(fieldnames(loadedSignals), 'sig_Kos_JurM_Dex_1_1_ch1'))
    %numSignals = 1929;
    
    % Iterate over each signal
    for i = 1:numSignals
        signal = signals{i}; % Signal is [channels x timePoints]
        windows = divideIntoWindows(signal, windowLength, samplingFreqs(i)); % Use new function
        numWindows = size(windows, 3);
        featSequence = zeros(numel(featNames), numWindows); % Preallocate feature matrix

        % Compute features for each window in parallel
        for j = 1:numWindows
            window = windows(:,:,j); % Extract each normalized window
            [featVals, featNames] = sigInspectComputeFeatures(window, featNames, samplingFreqs(i));
            featSequence(:, j) = featVals; % Store features
        end

        % Store the feature sequence
        X{i} = featSequence;

        % Store the labels
        Y{i} = categorical(labels{i});

        if ismember(i, progressMarks)
            fprintf('Progress: feature computed for %.0f%% signals \n', (i / numSignals) * 100);
        end
    end

    fprintf('Feature computation completed for %d signals.\n', numSignals);
end




