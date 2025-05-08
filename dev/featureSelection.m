function [selectedFeatures_FS, evalMetrics, svmModel] = featureSelection(X_fs, Y_fs, costMatrix, criteria, patientIds)
    % Define evaluation function based on chosen criterion
    switch lower(criteria)
        case 'f1'
            evalFunc = @(trainX, trainY, testX, testY) 1 - computeF1Score(trainX, trainY, testX, testY, costMatrix);
        case 'youden'
            evalFunc = @(trainX, trainY, testX, testY) 1 - computeYoudenIndex(trainX, trainY, testX, testY, costMatrix);
        case 'pr_auc'
            evalFunc = @(trainX, trainY, testX, testY) 1 - computePRCurveAUC(trainX, trainY, testX, testY, costMatrix);
        otherwise
            error('Invalid criterion. Choose from "f1", "youden", or "pr_auc".');
    end

    % Perform forward feature selection using 5-fold cross-validation
    opts = statset('display', 'iter');

    % Each patient has 10 samples
    expandedPatientIds = repelem(patientIds, 10);

    cv = cvpartition(expandedPatientIds, 'KFold', 5);  % Grouped by patient ID
    fprintf('Size of X_fs: %d rows\n', size(X_fs, 1));
    fprintf('Size of Y_fs: %d rows\n', length(Y_fs));
    fprintf('cv.NumObservations: %d\n', cv.NumObservations);

    

    [rankedFeatures, ~] = sequentialfs(evalFunc, X_fs, Y_fs, 'cv', cv, 'options', opts);

    % [rankedFeatures, ~] = sequentialfs(evalFunc, X_fs, Y_fs, 'cv', 5, 'options', opts);
    selectedFeatures_FS = find(rankedFeatures);

    % Train the final model on the full dataset with selected features
    svmModel = fitcsvm(X_fs(:, selectedFeatures_FS), Y_fs, 'KernelFunction', 'RBF', 'Cost', costMatrix, 'Standardize', true);

    cvSVM = crossval(svmModel, 'CVPartition', cv);

    % Compute evaluation metrics on the training data
    predictions = kfoldPredict(cvSVM);

    evalMetrics = computeEvaluationMetrics(Y_fs, predictions);
end
