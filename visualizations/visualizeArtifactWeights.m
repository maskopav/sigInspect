function patientSummary = visualizeArtifactWeights(X, Y, signalIds, artifactNames, colors)
    % plotArtifactHistogram plots histograms of artifact counts per patient,
    % overlays mean ± 2*std, and prints filtering stats.
    % 
    % Inputs:
    %   X, Y - data and labels
    %   signalIds - cell array of signal identifiers
    %   artifactNames - cell array of artifact type names (default: {'POW', 'BASE', 'FREQ'})
    %   colors - cell array of RGB triplets for coloring (default: {[0.2 0.6 1], [1.0 0.7 0.0], [0.8 0.2 0.2]})
    
    % Default parameter values
    if nargin < 4 || isempty(artifactNames)
        artifactNames = {'POW', 'BASE', 'FREQ'};
    end

    if nargin < 5 || isempty(colors)
        colors = {[0.2 0.6 1], [1.0 0.7 0.0], [0.8 0.2 0.2]};
    end
    
    % Extract artifact data for each type
    artifactData = cell(1, 3);
    for i = 1:3
        [~, artifactData{i}, signalIdsWindows] = extractFeatureValues(X, Y, i+1, signalIds);
    end
    
    % Get patient IDs from signalIds
    [patientIds, ~] = getPatientIds(signalIdsWindows);
    
    % Display patient distribution
    tabulate(patientIds);
    
    % Count artifacts per patient for each artifact type
    counts = cell(1, 3);
    for i = 1:3
        counts{i} = countArtifactPerPatient(artifactData{i}, patientIds);
    end
    
    % Extract all counts and patient IDs
    allPatientIds = unique(patientIds);
    n = length(allPatientIds);

    % Calculate total windows per patient
    totalWindows = zeros(n, 1);
    for i = 1:n
        totalWindows(i) = sum(strcmp(patientIds, allPatientIds{i}));
    end
    
    % Prepare data for plotting
    artifactCounts = zeros(n, 3);
    
    % Extract the artifact counts for all patients
    for i = 1:3
        for j = 1:length(counts{i}.patientIds)
            idx = find(strcmp(allPatientIds, counts{i}.patientIds{j}));
            if ~isempty(idx)
                artifactCounts(idx, i) = counts{i}.artifactCounts(j);
            end
        end
    end
    
    % Extract numeric patient IDs for sorting
    patientNum = str2double(regexprep(allPatientIds, 'sig_', ''));
    patientClusters = cellfun(@(id) id(1:3), allPatientIds, 'UniformOutput', false);

    for i = 1:length(patientClusters)
        if strcmp(patientClusters{i}, 'Bra')
            patientClusters{i} = 'BRATISLAVA';
        elseif strcmp(patientClusters{i}, 'Brn')
            patientClusters{i} = 'BRNO';
        elseif strcmp(patientClusters{i}, 'Olo')
            patientClusters{i} = 'OLOMOUC';
        else
            patientClusters{i} = 'UNKNOWN'; 
        end
    end

    % First sort by cluster name, then by patient number within each cluster
    uniqueClusters = unique(patientClusters);
    numClusters = length(uniqueClusters);
    
    % Create a new order based on clusters
    newOrder = [];
    clusterBoundaries = zeros(numClusters+1, 1);
    clusterBoundaries(1) = 1;
    
    for i = 1:numClusters
        cluster = uniqueClusters{i};
        % Find all patients in this cluster
        clusterPatientIdx = find(strcmp(patientClusters, cluster));
        
        % Sort patients within this cluster by patient number
        [~, sortIdxWithinCluster] = sort(patientNum(clusterPatientIdx));
        sortedClusterPatientIdx = clusterPatientIdx(sortIdxWithinCluster);
        
        % Add these patients to the new order
        newOrder = [newOrder; sortedClusterPatientIdx];
        
        % Record the boundary after this cluster
        clusterBoundaries(i+1) = clusterBoundaries(i) + length(sortedClusterPatientIdx);
    end
    
    % Reorder everything based on the new order
    patientIdsSorted = allPatientIds(newOrder);
    patientNumSorted = patientNum(newOrder);
    artifactCountsSorted = artifactCounts(newOrder, :);
    totalWindowsSorted = totalWindows(newOrder);
    patientClustersSorted = patientClusters(newOrder);
    
    % Calculate weights (percentage of each artifact type per patient)
    totalArtifacts = sum(artifactCountsSorted, 1);
    weights = 100 * artifactCountsSorted ./ totalArtifacts;
    
    % Plot the bar chart with switched axes (patients on y-axis)
    figure('Color', 'w', 'Position', [100, 100, 900, 1000]);
    
    % Create horizontal grouped bar chart
    hb = barh(weights, 'grouped');

    ax = gca;
    % Remove the box around the plot
    box(ax, 'off');

    % % Make major ticks point outward (optional, looks nicer)
    ax.TickDir = 'none';

    % Set bar colors
    for i = 1:3
        set(hb(i), 'FaceColor', colors{i});
    end
    
    % Add labels and grid
    title('Artifact weights per patient and center', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Patient ID', 'FontSize', 12);
    xlabel('Weight (% of artifact windows per patient)', 'FontSize', 12);
    
    % Set y-axis ticks to patient IDs
    yticks(1:n);
    yticklabels(1:n);

    % Add cluster dividers and labels
    hold on;
    for i = 2:length(clusterBoundaries)
        % Add horizontal line between clusters
        boundary = clusterBoundaries(i) - 0.5;
        if boundary < n
            line([0, 35], [boundary, boundary], 'Color', 'k', 'LineStyle', "-", 'LineWidth', 1);
            
            % Add cluster name annotation
            clusterName = uniqueClusters{i-1};
            midPoint = (clusterBoundaries(i-1) + clusterBoundaries(i) - 1) / 2;
            text(max(max(weights))*0.95, midPoint, clusterName, ...
                'FontWeight', 'bold', ...
                'HorizontalAlignment', 'left', ...
                'VerticalAlignment', 'middle', ...
                'FontSize', 10, ...
                'Color', 'k');
        end
    end
    
    % Add last cluster name
    if numClusters > 0
        midPoint = (clusterBoundaries(end-1) + n) / 2;
        text(max(max(weights))*0.95, midPoint, uniqueClusters{end}, ...
            'FontWeight', 'bold', ...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 10, ...
            'Color', 'k');
    end
     
    hold on;

    % Add threshold line
    threshold = 15;
    xline(threshold, 'r--', 'LineWidth', 1.5);
    text(threshold + 1, n * 0.98, '15% threshold', ...
        'Color', 'r', ...
        'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top');

    
    % Annotate values on bars and highlight if > threshold
    for i = 1:n
        for j = 1:3
            val = weights(i, j);
            if val > 0
                % Calculate position for text (adjust for horizontal bars)
                y = i + (j - 2) * 0.25; % shift label within group
                x = val + 0.5;
                
                % Determine text color based on threshold
                if val > threshold
                    color = 'r';
                else
                    color = 'k';
                end
                
                % Add text annotation
                text(x, y, sprintf('%.1f%% (%d)', val, artifactCountsSorted(i, j)), ...
                    'Color', color, ...
                    'FontSize', 9, ...
                    'Rotation', 0);
            end
        end
    end
    
    % Add legend
    legend(hb, artifactNames, 'Location', 'best');
    
    xlim([0, 35]);

    hold off

    % Create comprehensive patient summary structure
    patientSummary = struct(...
        'patientIds', {patientIdsSorted}, ...
        'totalWindows', totalWindowsSorted, ...
        'artifactCounts', artifactCountsSorted, ...
        'artifactTypes', {artifactNames}, ...
        'percentagesOfTotal', weights, ...
        'clusters', {patientClustersSorted} ...
    );
    
    % Print summary table for review
    fprintf('\nPatient Artifact Summary (by cluster):\n');
    fprintf('%-10s %-15s %-12s %-10s %-10s %-10s\n', 'Patient', 'Cluster', 'TotalWindows', artifactNames{1}, artifactNames{2}, artifactNames{3});
    fprintf('%-10s %-15s %-12s %-10s %-10s %-10s\n', '-------', '-------', '-----------', '--------', '--------', '--------');
    
    currentCluster = '';
    for i = 1:n
        if ~strcmp(currentCluster, patientClustersSorted{i})
            currentCluster = patientClustersSorted{i};
            fprintf('%-10s %-15s %-12s %-10s %-10s %-10s\n', ...
                '---', ['[ ' currentCluster ' ]'], '---', '---', '---', '---');
        end
        
        fprintf('%-10s %-15s %-12d %-10d %-10d %-10d\n', ...
            patientIdsSorted{i}, ...
            patientClustersSorted{i}, ...
            totalWindowsSorted(i), ...
            artifactCountsSorted(i, 1), ...
            artifactCountsSorted(i, 2), ...
            artifactCountsSorted(i, 3));
    end
    
    % Print totals row
    fprintf('%-10s %-15s %-12d %-10d %-10d %-10d\n', ...
        'TOTAL', ...
        '', ...
        sum(totalWindowsSorted), ...
        sum(artifactCountsSorted(:, 1)), ...
        sum(artifactCountsSorted(:, 2)), ...
        sum(artifactCountsSorted(:, 3)));
    
    % Also create individual histograms for each artifact type
    % figure('Color', 'w');
    % for i = 1:3
    %     subplot(3, 1, i);
    %     data = counts{i}.artifactCounts;
    %     binWidth = 15;
    %     histogram(data, 'BinWidth', binWidth, 'FaceColor', colors{i}, 'FaceAlpha', 0.6);
    %     title(['Artifact Type: ', artifactNames{i}]);
    %     xlabel('Number of artifacts per patient');
    %     ylabel('Number of patients');
    %     ylim([0 11]);
    % 
    %     % Outlier detection and reporting
    %     [maxVal, idxMax] = max(data);
    %     outlierPatient = counts{i}.patientIds{idxMax};
    % 
    %     % Count number of signals that belong to the outlier patient
    %     isSamePatient = strcmp(patientIds, outlierPatient);
    %     numSignals = sum(isSamePatient);
    % 
    %     fprintf('\n%s:\nOutlier patient: %s with %d artifacts and %d signals windows.\n', ...
    %         artifactNames{i}, outlierPatient, maxVal, numSignals);
    % end
    % 
    % % Create boxplot comparison
    % data_all = [counts{1}.artifactCounts(:); counts{2}.artifactCounts(:); counts{3}.artifactCounts(:)];
    % group_all = [repmat({artifactNames{1}}, length(counts{1}.artifactCounts), 1); ...
    %              repmat({artifactNames{2}}, length(counts{2}.artifactCounts), 1); ...
    %              repmat({artifactNames{3}}, length(counts{3}.artifactCounts), 1)];
    % 
    % figure('Color', 'w');
    % boxplot(data_all, group_all);
    % ylabel('Number of artifacts per patient');
    % title('Artifact distributions by type');
end

