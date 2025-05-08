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

    figure('Position',[0 0 600 850]);
    t = tiledlayout(2,1);
    t.TileSpacing = 'compact';
    t.Padding = 'compact';

    % Proportion plot
    ax1 = nexttile; hold on; %subplot(2,1,1); hold on;
    p1 = plot(1:numWindows, percentClean, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Clean (true)');
    p2 = plot(1:numWindows, percentArtifact, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Artifact (true)');
    xlabel('Windows');
    ylabel('Proportion');
    ylim([0 1]);
    xlim([1 10])
    title('Proportion of true labels per window');
    legend('Location', 'best');
    grid on;

    % Shaded plot of soft labels
    ax2 = nexttile; hold on; %subplot(2,1,2); hold on;
    s1 = shadedErrorBar(1:numWindows, avgClean, stdClean, 'lineProps', {'b', 'DisplayName', 'Clean (soft AVG ± STD)', 'LineWidth', 1.5});
    s2 = shadedErrorBar(1:numWindows, avgArtifact, stdArtifact, 'lineProps', {'r', 'DisplayName', 'Artifact (soft AVG ± STD)', 'LineWidth', 1.5});
    sc1 = scatter(xData(labels==0), yData(labels==0), 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Clean (true)');
    sc2 = scatter(xData(labels==1), yData(labels==1), 10, 'r', 'filled', 'MarkerFaceAlpha', 0.3, 'DisplayName', 'Artifact (true)');
    xlabel('Windows');
    ylabel('Soft label');
    ylim([0 1])
    xlim([1 10])
    title('Soft labels per window');
    lgd = legend('Location', 'northoutside');
    lgd.NumColumns = 2;
    grid on;

    % === Manually adjust bottom axis to be bigger ===
    pos1 = get(ax1, 'Position');
    pos2 = get(ax2, 'Position');
    
    % Shrink top plot vertically
    pos1(4) = pos1(4) * 0.6;
    set(ax1, 'Position', pos1);
    
    % Expand bottom plot vertically
    pos2(2) = pos1(2) - pos2(4)*0.1; % Move up a bit
    pos2(4) = pos2(4) * 1.4; % Make it taller
    set(ax2, 'Position', pos2);
    
    % === Combined Legend ===
    % lgd = legend([p1, p2, s1.mainLine, s2.mainLine, sc1, sc2]);
    % lgd.Layout.Tile = 'north'; % Still works in older MATLAB
    % lgd.FontSize = 9;
    % lgd.NumColumns = 2;
end
