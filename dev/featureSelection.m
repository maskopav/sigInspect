function [selectedFeatures_FS, accuracy, sensitivity, specificity, precision, f1, svmModel] = featureSelection(X_fs, Y_fs, costMatrix, criteria)
    % Define evaluation function based on chosen criterion
    switch lower(criteria)
        case 'f1'
            evalFunc = @(trainX, trainY, testX, testY) 1 - computeF1Score(trainX, trainY, testX, testY, costMatrix);
        case 'youden'
            evalFunc = @(trainX, trainY, testX, testY) 1 - computeYoudenIndex(trainX, trainY, testX, testY, costMatrix);
        case 'recall'
            evalFunc = @(trainX, trainY, testX, testY) 1 - computeRecall(trainX, trainY, testX, testY, costMatrix);
        otherwise
            error('Invalid criterion. Choose from "f1", "youden", or "balanced_accuracy".');
    end

    % Perform forward feature selection using 5-fold cross-validation
    opts = statset('display', 'iter');
    [rankedFeatures, ~] = sequentialfs(evalFunc, X_fs, Y_fs, 'cv', 5, 'options', opts);
    selectedFeatures_FS = find(rankedFeatures);

    % Train the final model on the full dataset with selected features
    svmModel = fitcsvm(X_fs(:, selectedFeatures_FS), Y_fs, 'KernelFunction', 'RBF', 'Cost', costMatrix);

    % Perform 5-fold cross-validation on the training data
    cvSVM = crossval(svmModel, 'KFold', 5);

    % Compute evaluation metrics on the training data
    predictions = kfoldPredict(cvSVM);

    [accuracy, sensitivity, specificity, precision, f1] = computeEvaluationMetrics(Y_fs, predictions);
end
