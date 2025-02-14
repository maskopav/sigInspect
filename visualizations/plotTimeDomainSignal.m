%% --- Function to plot time-domain signal ---
function plotTimeDomainSignal(signal, fs)
    t = (0:length(signal)-1) / fs;
    plot(t, signal, 'b');
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Time-Domain signal');
    grid on;
    hold on;
end
