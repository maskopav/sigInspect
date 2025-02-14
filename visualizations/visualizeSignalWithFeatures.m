function visualizeSignalWithFeatures(signal, fs, features, featureNames, winLength, useSpectrogram)
    % VISUALIZESIGNAL: Plots a signal in time and frequency domain,
    % with an option to use a spectrogram instead of FFT.
    %
    % Inputs:
    %   - signal:         1D array, time-domain signal
    %   - fs:             Sampling frequency (Hz)
    %   - features:       Matrix (NumFeatures x NumWindows), extracted features
    %   - featureNames:   Cell array (1 x NumFeatures), names of the features
    %   - winLength:      Length of each window in seconds
    %   - useSpectrogram: Boolean, true to plot spectrogram, false for FFT


    figure;

    % Plot time-domain signal
    subplot(3,1,1);
    plotTimeDomainSignal(signal, fs);

    % Compute and plot spectrogram or FFT
    subplot(3,1,2);
    if useSpectrogram
        computeAndPlotSpectrogram(signal, fs, true);
    else
        computeAndPlotFFT(signal, fs, true);
    end

    % Compute window centers for feature plotting
    t = (0:length(signal)-1) / fs;
    numWindows = floor(length(signal) / winLength);
    windowCenters = zeros(1, numWindows);

    for k = 1:numWindows
        startIdx = (k-1) * winLength + 1;
        endIdx = k * winLength;
        windowCenters(k) = mean(t(startIdx:endIdx));
    end

    % Plot features
    subplot(3,1,3);
    plotFeatures(windowCenters, features, featureNames);
end



