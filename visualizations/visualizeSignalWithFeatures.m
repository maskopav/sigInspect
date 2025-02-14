function visualizeSignalWithFeatures(signal, fs, windowSize, features, featureNames)
    % visualizeSignalWithFeatures Visualizes a signal in time and frequency domains
    % with overlaid windowed features (supporting multiple features).
    %
    % Parameters:
    %   signal       - Input signal vector.
    %   fs           - Sampling frequency in Hz.
    %   windowSize   - Size of each window in samples.
    %   features     - Matrix of feature values (NumberOfFeatures x NumberOfWindows).
    %   featureNames - (Optional) Cell array of feature names.

    % Validate feature names
    numFeatures = size(features, 1);
    if nargin < 5 || isempty(featureNames) || length(featureNames) ~= numFeatures
        featureNames = arrayfun(@(i) sprintf('Feature %d', i), 1:numFeatures, 'UniformOutput', false);
    end

    % Calculate time vector
    t = (0:length(signal)-1) / fs;

    % Number of windows
    numWindows = floor(length(signal) / windowSize);

    % Compute time centers of the windows
    featureTimes = zeros(1, numWindows);
    for k = 1:numWindows
        startIdx = (k-1) * windowSize + 1;
        endIdx = k * windowSize;
        featureTimes(k) = mean(t(startIdx:endIdx));
    end

    % Define distinct colors for each feature
    colors = lines(numFeatures);

    % Plot time-domain signal with overlaid features
    figure;
    subplot(3,1,1);
    plot(t, signal, 'k', 'LineWidth', 1); % Plot signal in black
    % hold on;
    % for i = 1:numFeatures
    %     plot(featureTimes, features(i,:), 'o-', 'Color', colors(i,:), 'LineWidth', 1.5);
    % end
    % hold off;
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Time-Domain signal');
    % legend(['Signal', featureNames]);

    % Plot frequency-domain signal
    subplot(3,1,2);
    nfft = 2^nextpow2(length(signal));
    f = fs/2 * linspace(0,1,nfft/2+1);
    Y = fft(signal, nfft) / length(signal);
    plot(f, 2*abs(Y(1:nfft/2+1)), 'b', 'LineWidth', 1);
    xlabel('Frequency (Hz)');
    ylabel('Magnitude');
    title('Frequency-Domain signal');

    % Plot multiple feature values over time
    subplot(3,1,3);
    hold on;
    for i = 1:numFeatures
        plot(featureTimes, features(i,:), 'o-', 'Color', colors(i,:), 'LineWidth', 1.5);
    end
    hold off;
    xlabel('Time (s)');
    ylabel('Feature Value');
    title('Windowed features over time');
    legend(featureNames);
end
