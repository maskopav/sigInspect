function [AUC_values, selectedFeatures_AUC] = computeAUC(X, Y, artifactIdx, numTopFeatures, featNames)
    % Extract feature values and labels
    [allFeatureValues, labels] = extractFeatureValues(X, Y, artifactIdx);
    
    numFeatures = size(allFeatureValues, 1);
    AUC_values = zeros(numFeatures, 1);
    
    % Standardize feature values
    allFeatureValues = (allFeatureValues - mean(allFeatureValues, 2)) ./ std(allFeatureValues, 0, 2);
    allFeatureValues(isnan(allFeatureValues)) = 0;
    
    % Compute AUC for each feature
    for featIdx = 1:numFeatures
        [~, ~, ~, AUC] = perfcurve(labels, allFeatureValues(featIdx, :), 1);
        AUC_values(featIdx) = AUC;
    end

    % Rank features based on AUC deviation from 0.5
    [~, sortedIdx] = sort(abs(AUC_values - 0.5), 'descend');
    selectedFeatures_AUC = sortedIdx(1:numTopFeatures);
    
    disp("Selected Top Features:");
    disp(featNames(selectedFeatures_AUC));
end
