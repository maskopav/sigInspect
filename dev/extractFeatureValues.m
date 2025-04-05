function [allFeatureValues, labels] = extractFeatureValues(X, Y, artifactIdx)
    % Features (rows: samples, cols: features)
    % Labels (0 = clean, 1 = artifact)
    numFeatures = size(X{1}, 1);
    totalSamples = sum(cellfun(@(x) size(x, 2), X)); % Total number of samples
    allFeatureValues = zeros(totalSamples, numFeatures); 
    labels = zeros(totalSamples, 1); % Labels vector
    
    colIdx = 1; % Column index tracker
    
    for i = 1:length(X)
        featVals = X{i}; % Feature values for this sample (numFeatures × numSamples)
        labelVals = Y{i}(artifactIdx, :); % Corresponding labels
        
        numSamples = size(featVals, 2); % Number of samples in this subset
        
        % Insert data into preallocated matrix at the correct positiona
        allFeatureValues(colIdx:colIdx+numSamples-1, :) = featVals';
        labels(colIdx:colIdx+numSamples-1) = labelVals;
        
        colIdx = colIdx + numSamples; % Update index
    end
    labels = categorical(labels); 
end
