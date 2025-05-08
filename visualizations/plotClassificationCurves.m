function plotClassificationCurves(predictedProbsCell, labelsCell, plotType, varargin)
% Plot ROC or PR curves for up to N datasets
%
% predictedProbsCell: cell of predicted probabilities (1xN), each element
% can be a vector or a cell array of vectors (e.g., cross-validation folds)
% labelsCell: cell of true binary labels (1xN), same structure as predictedProbsCell
% plotType: 'roc' or 'pr'
% Optional name-value pairs:
%   'Threshold'     — scalar threshold to mark on curves (default = [])
%   'DatasetNames'  — cell of names for legend (default = {'Dataset 1', ...})
%   'Colors'        — cell of RGB triplets for colors

p = inputParser;
addParameter(p, 'Threshold', []);
addParameter(p, 'DatasetNames', arrayfun(@(i) sprintf('Dataset %d',i), 1:numel(predictedProbsCell), 'UniformOutput', false));
addParameter(p, 'Colors', {[0 0 1], [0.85 0.33 0.1], [0.49 0.18 0.56], [0.3 0.75 0.93]}); % blue, orange-red, purple, cyan
parse(p, varargin{:});

threshold = p.Results.Threshold;
datasetNames = p.Results.DatasetNames;
colors = p.Results.Colors;

% === Plot ===
figure('Position',[0 0 600 900]); hold on;
curveHandles = gobjects(1, numel(predictedProbsCell));  % store line handles
legendEntries = cell(1, numel(predictedProbsCell));
thresholdHandle = [];

for i = 1:numel(predictedProbsCell)
    probs = predictedProbsCell{i};
    labels = labelsCell{i};
    
    % --- Flatten and concatenate regardless of vector lengths ---
    if iscell(probs)
        probs = cellfun(@(x) x(:), probs, 'UniformOutput', false);  % force column
        probs = cat(1, probs{:});
    else
        probs = probs(:);
    end
    
    if iscell(labels)
        labels = cellfun(@(x) x(:), labels, 'UniformOutput', false);  % force column
        labels = cat(1, labels{:});
    else
        labels = labels(:);
    end
    
    % === Compute curves and AUC ===
    if strcmpi(plotType, 'roc')
        [x, y, thresholds, auc] = perfcurve(labels, probs, 1);
        xlab = 'False Positive Rate';
        ylab = 'True Positive Rate';
    elseif strcmpi(plotType, 'pr')
        [x, y, thresholds, auc] = perfcurve(labels, probs, 1, 'xCrit','reca','yCrit','prec');
        xlab = 'Recall';
        ylab = 'Precision';
    else
        error('Invalid plotType. Use ''roc'' or ''pr''.');
    end
    
    % Plot curve and store handle
    if i == 1
        curveHandles(i) = plot(x, y, 'LineWidth', 2, 'Color', colors{i});
    else 
        curveHandles(i) = plot(x, y, ":", 'LineWidth', 2, 'Color', colors{i});
    end
    
    
    % If threshold is specified, plot marker
    if ~isempty(threshold)
        [~, idx] = min(abs(thresholds - threshold));
        hTemp = plot(x(idx), y(idx), 'kx', 'MarkerSize', 10, 'LineWidth', 2);
        % Only save one handle for legend
        if isempty(thresholdHandle)
            thresholdHandle = hTemp;
        end
    end
    
    legendEntries{i} = sprintf('%s (AUC = %.2f)', datasetNames{i}, auc);
end

xlabel(xlab); ylabel(ylab);
%title(upper(plotType) + " Curves");

% === Add threshold entry to legend if needed ===
if ~isempty(threshold)
    thresholdLegend = sprintf('Threshold = %.2f', threshold);
    legend([curveHandles thresholdHandle], [legendEntries thresholdLegend], 'Location','Best');
else
    legend(curveHandles, legendEntries, 'Location','Best');
end

grid off; box off;
end
