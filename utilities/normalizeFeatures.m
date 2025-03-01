function normFeatures = normalizeFeatures(features)
    % NORMALIZEFEATURES Normalizes feature matrix to [0,1] range
    %   features - Input feature matrix (numFeatures x numWindows), normalize each feature across all signals
    %   normFeatures - Normalized features
    
    minVals = min(features, [], 2); % Min per feature and signal
    maxVals = max(features, [], 2); % Max per feature and signal
    rangeVals = maxVals - minVals;
    rangeVals(rangeVals == 0) = 1; % Prevent NaN

    normFeatures = (features - minVals) ./ rangeVals;


end
