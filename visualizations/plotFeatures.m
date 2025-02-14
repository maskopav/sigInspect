function plotFeatures(windowCenters, features, featureNames)
    % Ensure featureNames matches the number of feature rows
    numFeatures = size(features, 1);
    
    if length(featureNames) ~= numFeatures
        warning('Number of feature names does not match number of features. Truncating/ignoring extras.');
        featureNames = featureNames(1:min(length(featureNames), numFeatures));
    end

    % Generate distinct colors for each feature
    colors = lines(numFeatures);  

    % Plot features with unique colors
    hold on;
    for i = 1:numFeatures
        % Remove NaNs to avoid plotting issues
        validIdx = ~isnan(features(i, :));
        plot(windowCenters(validIdx), features(i, validIdx), '-o', 'Color', colors(i, :), 'LineWidth', 1.5);
    end
    
    xlabel('Time (s)');
    ylabel('Feature Value');
    title('Windowed Features Over Time');
    
    % Only assign legend if names exist
    if ~isempty(featureNames)
        legend(featureNames, 'Location', 'best');
    end
    
    hold off;
end
