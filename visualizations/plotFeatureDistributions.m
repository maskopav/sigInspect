function plotFeatureDistributions(AUC_values, allFeatureValues, labels, featNames, selectedFeatures_AUC, artifactIdx, numTopFeatures)
    figure; hold on;
    colors = lines(numTopFeatures);
    legendEntries = cell(2*numTopFeatures, 1);
    
    for i = 1:numTopFeatures
        featIdx = selectedFeatures_AUC(i);
        topFeatureValues = allFeatureValues(featIdx, :);
        
        feature_0 = topFeatureValues(labels == 0);
        feature_1 = topFeatureValues(labels == 1);
        
        if abs(skewness(topFeatureValues)) > 2
            min_value = min([feature_0, feature_1]);
            if min_value <= 0
                feature_0 = log1p(feature_0 - min_value + 0.01);
                feature_1 = log1p(feature_1 - min_value + 0.01);
            else
                feature_0 = log1p(feature_0);
                feature_1 = log1p(feature_1);
            end
        end
        
        [pdf_0, x_0] = ksdensity(feature_0);
        [pdf_1, x_1] = ksdensity(feature_1);
        
        plot(x_0, pdf_0, '-', 'Color', colors(i, :), 'LineWidth', 2);
        plot(x_1, pdf_1, '--', 'Color', colors(i, :), 'LineWidth', 2);
        
        legendEntries{(2*i-1)} = sprintf('%s 0 - AUC: %.2f', featNames{featIdx}, AUC_values(featIdx));
        legendEntries{(2*i)} = sprintf('%s 1 - AUC: %.2f', featNames{featIdx}, AUC_values(featIdx));
    end
    
    legend(legendEntries, 'Location', 'Best');
    xlabel('Feature value');
    ylabel('Probability density');
    title(sprintf('PDFs of %d top features by AUC, Artifact %d', numTopFeatures, artifactIdx));
    hold off;
end


