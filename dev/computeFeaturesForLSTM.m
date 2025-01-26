function [X, Y, featNames] = computeFeaturesForLSTM(signals, labels, windowLength, samplingFreq, featNames)
    % INPUTS:
    % signals: Cell array of signals, each signal is [channels x timePoints].
    % labels: Cell array of windows annotation for each signal (1 for artifact, 0 for non-artifact) [1 x windows].
    % windowLength: Length of each window (in seconds).
    % samplingFreq: Sampling frequency of the signals.
    % featNames: Names of features to compute for each window.
    %
    % OUTPUTS:
    % X: Cell array where each cell is a matrix [timeSteps x features].
    % Y: Categorical labels for each window (binary).

    numSignals = numel(signals);
    X = cell(numSignals, 1);
    Y = cell(numSignals, 1);

    % Compute number of samples per window
    segmentLength = windowLength * samplingFreq;

    % Progress tracking
    fprintf('Computing features for %d signals...\n', numSignals);

    progressMarks = round([0.01, 0.1, 0.25, 0.50, 0.75, 1] * numSignals);

    % Iterate over each signal
    for i = 1:numSignals
        signal = signals{i}; % Signal is [channels x timePoints]
        numWindows = floor(size(signal, 2) / segmentLength);
        featSequence = zeros(numel(featNames), numWindows); % Preallocate feature matrix for signal
        
        % Extract and process windows in vectorized form
        windows = reshape(signal(:, 1:numWindows * segmentLength), size(signal, 1), segmentLength, numWindows);
        
        % Compute features for each window in parallel
        for j = 1:numWindows
            window = windows(:,:,j); % Extract each normalized window
            [featVals, featNames] = sigInspectComputeFeatures(window, featNames, samplingFreq);
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




