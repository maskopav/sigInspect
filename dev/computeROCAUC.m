function rocAUC = computeROCAUC(labels, scores, positiveClass)
% COMPUTEROCAUC Computes the area under the ROC curve (AUC-ROC)
%
% Inputs:
%   labels - Ground truth labels
%   scores - Prediction scores (probabilities) for the positive class
%   positiveClass - Value that represents the positive class (default: 1)
%
% Output:
%   rocAUC - Area under the ROC curve (AUC-ROC)

    % Set default for positive class if not provided
    if nargin < 3
        positiveClass = 1;
    end

    % Compute false positive rate (FPR) and true positive rate (TPR)
    try
        [fpr, tpr, ~, rocAUC] = perfcurve(labels, scores, positiveClass);
    catch e
        warning('Error in perfcurve: %s', e.message);
        rocAUC = NaN;
        return;
    end

    % Handle edge case: If too few points to compute ROC curve
    if length(fpr) < 2 || length(tpr) < 2
        warning('Too few points to compute a meaningful ROC curve');
        rocAUC = NaN;
        return;
    end

    % Find valid indices (non-NaN values)
    validIndices = ~isnan(fpr) & ~isnan(tpr);

    % Check if we have enough valid points
    if sum(validIndices) < 2
        warning('Too few valid points to compute ROC AUC (after removing NaNs)');
        rocAUC = NaN;
        return;
    end

    % Extract valid FPR and TPR values
    validFPR = fpr(validIndices);
    validTPR = tpr(validIndices);

    % Sort by FPR for proper integration (although perfcurve usually does this)
    [sortedFPR, sortIdx] = sort(validFPR);
    sortedTPR = validTPR(sortIdx);

    % Add endpoints if they don't exist (to ensure curve starts at (0,0) and ends at (1,1))
    if sortedFPR(1) > 0
        sortedFPR = [0; sortedFPR];
        sortedTPR = [0; sortedTPR];
    end

    if sortedFPR(end) < 1
        sortedFPR = [sortedFPR; 1];
        sortedTPR = [sortedTPR; 1];
    end

    % Compute area under ROC curve using trapezoidal rule (in case built-in AUC is NaN)
    computedAUC = trapz(sortedFPR, sortedTPR);

    % Prefer perfcurve’s built-in AUC if valid, else fallback to computed one
    if isnan(rocAUC) || rocAUC < 0
        rocAUC = computedAUC;
    end

    % Handle edge case where AUC might be slightly negative due to numerical issues
    if rocAUC < 0 && rocAUC > -1e-10
        rocAUC = 0;
    end
end
