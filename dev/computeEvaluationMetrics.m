function metrics = computeEvaluationMetrics(labels, predictions)
    % Convert both labels and predictions to categorical or numeric type
    if iscategorical(labels)
        labels = double(string(labels)); % Convert categorical labels to numeric
    end
    if iscategorical(predictions)
        predictions = double(string(predictions));
    end

    % Ensure predictions are binary (0 or 1)
    uniquePreds = unique(predictions);
    if any(uniquePreds ~= 0 & uniquePreds ~= 1)
        error("Predictions contain non-binary values: %s", mat2str(uniquePreds));
    end

    metrics = struct();

    % Ensure confusion matrix has correct class order: 0 (negative), 1 (positive)
    confMat = confusionmat(labels, predictions, 'Order', [0 1]);

    % Extract confusion matrix elements
    TN = confMat(1,1); FP = confMat(1,2);
    FN = confMat(2,1); TP = confMat(2,2);

    % Compute metrics
    accuracy = (TP + TN) / sum(confMat(:));
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    precision = TP / (TP + FP);
    recall = sensitivity;
    f1 = 2 * (precision * recall) / (precision + recall);

    % Handle NaN cases
    sensitivity(isnan(sensitivity)) = 0;
    specificity(isnan(specificity)) = 0;
    precision(isnan(precision)) = 0;
    f1(isnan(f1)) = 0;

    metrics.accuracy = accuracy;
    metrics.sensitivity = sensitivity;
    metrics.specificity = specificity;
    metrics.precision = precision;
    metrics.f1 = f1;
    metrics.youden = metrics.sensitivity + metrics.specificity - 1;
end
