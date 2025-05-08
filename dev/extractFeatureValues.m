function [allFeatureValues, labels, sigIdsValues] = extractFeatureValues(X, Y, artifactIdx, signalIds)
    % Features (rows: samples, cols: features)
    % Labels (0 = clean, 1 = artifact)
    numFeatures = size(X{1}, 1);
    totalSamples = sum(cellfun(@(x) size(x, 2), X)); % Total number of samples
    allFeatureValues = zeros(totalSamples, numFeatures); 
    labels = zeros(totalSamples, 1); % Labels vector
    sigIdsValues = cell(totalSamples, 1); 
    
    colIdx = 1; % Column index tracker
    
    for i = 1:length(X)
        featVals = X{i}; % Feature values for this sample (numFeatures × numSamples)
        labelVals = Y{i}(artifactIdx, :); % Corresponding labels
        sigVal = signalIds{i};
        
        numSamples = size(featVals, 2); % Number of samples in this subset
        
        % Insert data into preallocated matrix at the correct positiona
        allFeatureValues(colIdx:colIdx+numSamples-1, :) = featVals';

        % disp(size(labels(colIdx:colIdx+numSamples-1)))
        % disp(size(labelVals))
        labels(colIdx:colIdx+numSamples-1) = labelVals;
        
        sigIdsValues(colIdx:colIdx+numSamples-1) = repmat({sigVal}, numSamples, 1);
        
        colIdx = colIdx + numSamples; % Update index
    end
    % labels = categorical(labels); 
end
