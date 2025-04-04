function [allFeatureValues, labels] = extractFeatureValues(X, Y, artifactIdx)
    numFeatures = size(X{1}, 1);
    totalSamples = sum(cellfun(@(x) size(x, 2), X)); % Total number of samples
    allFeatureValues = zeros(numFeatures, totalSamples); 
    labels = zeros(1, totalSamples); % Labels vector
    
    colIdx = 1; % Column index tracker
    
    for i = 1:length(X)
        featVals = X{i}; % Feature values for this sample (numFeatures × numSamples)
        labelVals = Y{i}(artifactIdx, :); % Corresponding labels
        
        numSamples = size(featVals, 2); % Number of samples in this subset
        
        % Insert data into preallocated matrix at the correct position
        allFeatureValues(:, colIdx:colIdx+numSamples-1) = featVals;
        labels(colIdx:colIdx+numSamples-1) = labelVals;
        
        colIdx = colIdx + numSamples; % Update index
    end
end
