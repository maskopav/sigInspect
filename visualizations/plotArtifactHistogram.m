function counts = plotArtifactHistogram(X, Y, signalIds)
    % plotArtifactHistogram plots histograms of artifact counts per patient,
    % overlays mean ± 2*std, and prints filtering stats.
    
    % signalIds would be the same for all artifacts..
    [~, artifact2, signalIdsWindows] = extractFeatureValues(X, Y, 2, signalIds);% POWER
    [~, artifact3, ~] = extractFeatureValues(X, Y, 3, signalIds); % BASE
    [~, artifact4, ~] = extractFeatureValues(X, Y, 4, signalIds); % FREQ

    % Get patients Ids from signalIds
    [patientIds, ~] = getPatientIds(signalIdsWindows);
    
    tabulate(patientIds)

    % Count artifacts per patient for each artifact type
    counts{1} = countArtifactPerPatient(artifact2, patientIds);
    counts{2} = countArtifactPerPatient(artifact3, patientIds);
    counts{3} = countArtifactPerPatient(artifact4, patientIds);

    disp(size(counts{1}))

    labels = {'Artifact Type 1 (Power)', 'Artifact Type 2 (Base)', 'Artifact Type 3 (Freq)'};
    colors = {[0.1, 0.5, 0.8], [0.93, 0.69, 0.13], [0.9, 0.4, 0.4]};
    
    figure('Color', 'w'); hold on;
    legendEntries = cell(1, 3);
    binWidth = 15;
    for i = 1:3
        subplot(3, 1, i); 
        data = counts{i}.artifactCounts;
        histogram(data, 'BinWidth', binWidth, 'FaceColor', colors{i}, 'FaceAlpha', 0.6);
        title(labels{i});
        xlabel('Number of artifacts per patient');
        ylabel('Number of patients');
        ylim([0 11]);
    
        % --- OUTLIER DETECTION ---
        [maxVal, idxMax] = max(data);
        outlierPatient = counts{i}.patientIds{idxMax};
    
        % Count number of signals that belong to the outlier patient
        isSamePatient = strcmp(patientIds, outlierPatient);
        numSignals = sum(isSamePatient);
    
        fprintf('\n%s:\nOutlier patient: %s with %d artifacts and %d signals windows.\n', ...
        labels{i}, outlierPatient, maxVal, numSignals);
        
        % Fit and plot PDF
        % data = counts{i};
        % data = data(~isnan(data) & data > 0);
        % pd = fitdist(data(:), 'Lognormal');
        % hold on
        % x_vals = 0:binWidth:max(data);
        % y = pdf(pd, x_vals);
        % yyaxis right
        % plot(x_vals, y, 'Color', colors{i}, 'LineWidth', 1.5);
        % ylabel('Probability Density');

        % % Remove NaNs or negatives (if any)
        % data = counts{i};
        % data = data(~isnan(data) & data > 0);
        % shift = 0;  % or any small constant to avoid log(0)
        % data_shifted = data + shift;
        % % Fit log-normal distribution
        % pd = fitdist(data_shifted(:), 'Lognormal');
        % 
        % x_vals = 0:binWidth:max(data);
        % pdf_vals = pdf(pd, x_vals + shift);
        % 
        % % Plot histogram
        % h = histogram(data, 'BinWidth', binWidth, ...
        %     'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
        %     'FaceColor', colors{i}, 'DisplayName', labels{i});
        % 
        % % Overlay PDF
        % yyaxis right
        % plot(x_vals, pdf_vals, 'Color', colors{i}, 'LineWidth', 1.5);
        % yyaxis left
        % 
        % % Mark 95th percentile threshold
        % q95 = prctile(data, 95);
        % xline(q95, '--', 'Color', colors{i}, 'LineWidth', 2);
        % 
        % % Store legend entry
        % legendEntries{i} = sprintf('%s: μ=%.1f, σ=%.1f, 95%%=%.0f', ...
        %     labels{i}, mean(data), std(data), q95);
        % 
        % fprintf('%s: %d / %d patients above 95%% (outliers)\n', ...
        %     labels{i}, sum(data > q95), numel(data));

    end

    % xlabel('Number of artifacts per patient');
    % ylabel('Number of patients');
    % title('Artifact distribution per patient by type');
    % legend(legendEntries, 'Location', 'northeast');
    grid on; box on;


    data_all = [counts{1}.artifactCounts(:); counts{2}.artifactCounts(:); counts{3}.artifactCounts(:)];
    group_all = [repmat({'Power'}, length(counts{1}.artifactCounts), 1); ...
                 repmat({'Base'}, length(counts{2}.artifactCounts), 1); ...
                 repmat({'Freq'}, length(counts{3}.artifactCounts), 1)];
    
    figure;
    boxplot(data_all, group_all);
    ylabel('Number of artifacts per patient');
    title('Artifact distributions by type');

end

