function windows = divideIntoWindows(signal, windowLength, samplingFreq)
    % INPUTS:
    % signal: Matrix of size [channels x timePoints]
    % windowLength: Length of each window in seconds
    % samplingFreq: Sampling frequency of the signal
    %
    % OUTPUT:
    % windows: 3D matrix of size [channels x segmentLength x numWindows]

    segmentLength = windowLength * samplingFreq; % Compute number of samples per window
    numWindows = floor(size(signal, 2) / segmentLength); % Calculate number of full windows
    
    % Reshape signal into windows
    windows = reshape(signal(:, 1:numWindows * segmentLength), size(signal, 1), segmentLength, numWindows);
end