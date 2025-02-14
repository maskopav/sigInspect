function [f, Y] = computeAndPlotFFT(signal, fs, plotFlag)
    L = length(signal);
    f = fs*(0:(L/2))/L;
    Y = abs(fft(signal)/L);
    Y = Y(1:length(f)); % Take positive frequencies

    if plotFlag
        plot(f, Y, 'r');  
        xlabel('Frequency (Hz)');
        ylabel('Magnitude');
        title('Frequency-Domain (FFT)');
        grid on;
    end
end