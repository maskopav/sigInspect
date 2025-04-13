function plotSoftLabelDistributions(predictedProbs, Y)
    % Per-window analysis
    allProbsWindows = cellfun(@(x) x(2, :), predictedProbs, 'UniformOutput', false);
    allProbsWindows = vertcat(allProbsWindows{:}); 

    allTrueLabels = cellfun(@(x) x(:)', Y, 'UniformOutput', false);
    allTrueLabels = vertcat(allTrueLabels{:}); 
    allTrueLabels = double(string(allTrueLabels));

    [numSamples, numWindows] = size(allProbsWindows);
    [winX, ~] = meshgrid(1:numWindows, 1:numSamples);
    xData = winX(:);
    yData = allProbsWindows(:);
    labels = allTrueLabels(:);

    cmap = [0 0 1; 1 0 0];  % blue = clean, red = artifact
    colorData = cmap(labels + 1, :);

    probsCleanMask = allProbsWindows;
    probsCleanMask(allTrueLabels == 1) = NaN;
    probsArtifMask = allProbsWindows;
    probsArtifMask(allTrueLabels == 0) = NaN;

    avgClean = mean(probsCleanMask, 1, 'omitnan');
    stdClean = std(probsCleanMask, 0, 1, 'omitnan');
    avgArtifact = mean(probsArtifMask, 1, 'omitnan');
    stdArtifact = std(probsArtifMask, 0, 1, 'omitnan');

    % Proportion of true labels per window
    percentClean = sum(allTrueLabels == 0, 1) ./ numSamples;
    percentArtifact = sum(allTrueLabels == 1, 1) ./ numSamples;

    % figure
    % Proportion plot
    subplot(2,1,1); hold on;
    plot(1:numWindows, percentClean, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Clean (true)');
    plot(1:numWindows, percentArtifact, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Artifact (true)');
    xlabel('Windows');
    ylabel('Proportion');
    ylim([0 1]);
    title('Proportion of true labels per window');
    legend('Location', 'best');
    grid on;

    % Shaded plot of soft labels
    subplot(2,1,2); hold on;
    shadedErrorBar(1:numWindows, avgClean, stdClean, 'lineProps', {'b', 'DisplayName', 'Clean avg ± STD'});
    shadedErrorBar(1:numWindows, avgArtifact, stdArtifact, 'lineProps', {'r', 'DisplayName', 'Artifact avg ± STD'});
    scatter(xData(labels==0), yData(labels==0), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
    scatter(xData(labels==1), yData(labels==1), 10, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
    xlabel('Windows');
    ylabel('Soft label');
    ylim([0 1])
    title('Soft labels per window');
    legend('Location', 'best');
    grid on;
end
