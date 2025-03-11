function [numPeaks, meanPeakHeight, peakFreq, peakRMSRatio, avgPeakWidth] = compPowerPeakFeatures(signal, fs, smoothWindowDuration, plotOption)
    powerSignal = signal.^2;

    % Gaussian smoothing preserves peak structure better than a MA
    windowSize = round(smoothWindowDuration * fs);
    gaussKernel = gausswin(windowSize) / sum(gausswin(windowSize)); 
    if length(powerSignal) < 3 * windowSize
        smoothSignal = conv(powerSignal, gaussKernel, 'same');
    else
        smoothSignal = filtfilt(gaussKernel, 1, powerSignal);
    end

    % Adaptive thresholding based on signal variation
    baselineNoise = median(abs(smoothSignal - median(smoothSignal))); % Robust estimate
    minProm = 1.5 * baselineNoise;  % Adaptive threshold
    minHeight = median(smoothSignal) + 8 * baselineNoise;
    minDistance = round(fs * 0.004); %s

    % Initialize the output variables in case signal is too short
    numPeaks = 0;
    meanPeakHeight = 0;
    peakFreq = 0;
    peakRMSRatio = 0;
    avgPeakWidth = 0;

    if length(smoothSignal) <= minDistance
        % Signal is too short, return default values
        disp('Signal too short for peak detection. Skipping findpeaks.');
        return; % No need to continue further
    else
        % Perform peak detection
        [pks, locs, width, ~] = findpeaks(smoothSignal, ...
            'MinPeakProminence', minProm, ...
            'MinPeakHeight', minHeight, ...
            'MinPeakDistance', minDistance);

        % Handle case where no peaks are found
        if isempty(pks)
            pks = NaN;
            locs = NaN;
        end

        % If peaks are found, calculate features
        if ~isempty(pks)
            numPeaks = numel(pks);
            meanPeakHeight = mean(pks, 'omitnan'); 
            avgPeakWidth = mean(width, 'omitnan');
            if numPeaks > 1
                peakIntervals = diff(locs) / fs; 
                peakFreq = 1 / mean(peakIntervals);
            end
        end
    end
    
    % Calculate peak RMS ratio (even if no peaks)
    peakRMSRatio = meanPeakHeight / rms(smoothSignal);

    % Optional: Plotting (if requested)
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
