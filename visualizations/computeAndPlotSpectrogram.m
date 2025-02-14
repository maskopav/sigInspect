%% --- Function to compute & optionally plot spectrogram ---
function [S, F, T, P] = computeAndPlotSpectrogram(signal, fs, plotFlag)
    NFFT = 1024; 
    nOverlap = round(3 * NFFT / 4);
    [S, F, T, P] = spectrogram(signal, NFFT, nOverlap, NFFT, fs);

    if plotFlag
        imagesc(T, F, log(abs(P)));
        set(gca, 'YDir', 'normal'); % Flip Y-axis so low freq at bottom
        ylim([0 3000]); % Limit to 0-3000 Hz
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title('Spectrogram');
        colorbar;
    end
end