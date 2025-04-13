function prAUC = computePRCurveAUC(labels, scores, positiveClass)
    % COMPUTEPRAUC Computes the area under the precision-recall curve
    %
    % Inputs:
    %   labels - Ground truth labels
    %   scores - Prediction scores (probabilities) for the positive class
    %   positiveClass - Value that represents the positive class (default: 1)
    %
    % Output:
    %   prAUC - Area under the precision-recall curve
    
    % Set default for positive class if not provided
    if nargin < 3
        positiveClass = 1;
    end
    
    % Compute precision and recall at various thresholds
    try
        [precision, recall, ~] = perfcurve(labels, scores, positiveClass, 'XCrit', 'reca', 'YCrit', 'prec');
    catch e
        warning('Error in perfcurve: %s', e.message);
        prAUC = NaN;
        return;
    end
    
    % Handle edge case: If too few points to compute PR curve
    if length(precision) < 2 || length(recall) < 2
        warning('Too few points to compute a meaningful PR curve');
        prAUC = NaN;
        return;
    end
    
    % Find valid indices (non-NaN values)
    validIndices = ~isnan(recall) & ~isnan(precision);
    
    % Check if we have enough valid points
    if sum(validIndices) < 2
        warning('Too few valid points to compute PR AUC (after removing NaNs)');
        prAUC = NaN;
        return;
    end
    
    % Extract valid precision and recall values
    validRecall = recall(validIndices);
    validPrecision = precision(validIndices);
    
    % Sort by recall for proper integration
    [sortedRecall, sortIdx] = sort(validRecall);
    sortedPrecision = validPrecision(sortIdx);
    
    % Add endpoints if they don't exist to ensure proper curve
    % Add (0,1) if not present (theoretical start of PR curve)
    if sortedRecall(1) > 0
        sortedRecall = [0; sortedRecall];
        sortedPrecision = [1; sortedPrecision];
    end
    
    % Make sure we have a point at recall=1 
    if sortedRecall(end) < 1
        % Use the last valid precision value as an approximation for recall=1
        sortedRecall = [sortedRecall; 1];
        sortedPrecision = [sortedPrecision; sortedPrecision(end)];
    end
    
    % Compute area under PR curve using trapezoidal rule
    prAUC = trapz(sortedRecall, sortedPrecision);
    
    % Handle edge case where AUC might be slightly negative due to numerical issues
    if prAUC < 0 && prAUC > -1e-10
        prAUC = 0;
    end
end