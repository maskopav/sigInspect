function [AUC_values, selectedFeatures_AUC, allFeatureValues, labels] = computeAUC(X, Y, artifactIdx, numTopFeatures, featNames)
    numFeatures = size(X{1}, 1);
    AUC_values = zeros(numFeatures, 1);
    allFeatureValues = [];
    
    for featIdx = 1:numFeatures
        featureValues = [];
        labels = [];
        
        for i = 1:length(X)
            featVals = X{i}(featIdx, :);
            labelVals = Y{i}(artifactIdx, :);
            cleanMask = Y{i}(1, :) == 1;
            %labelVals(~cleanMask & labelVals == 0) = NaN;
            
            featureValues = [featureValues, featVals];
            labels = [labels, labelVals];
        end
        
        featureValues = (featureValues - mean(featureValues)) / std(featureValues);
        featureValues(isnan(featureValues)) = 0;
        
        [~, ~, ~, AUC] = perfcurve(labels, featureValues, 1);
        AUC_values(featIdx) = AUC;
        
        allFeatureValues = [allFeatureValues; featureValues];
    end
    
    [~, sortedIdx] = sort(abs(AUC_values - 0.5), 'descend');
    selectedFeatures_AUC = sortedIdx(1:numTopFeatures);
    disp("Selected Top Features:");
    disp(featNames(selectedFeatures_AUC));
end