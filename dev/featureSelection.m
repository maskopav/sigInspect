function [selectedFeatures_FS, accuracy, sensitivity, specificity, precision, f1_score] = featureSelection(X_fs, Y_fs, costMatrix)    
    cv = cvpartition(Y_fs, 'HoldOut', 0.2);
    X_train = X_fs(training(cv), :);
    Y_train = Y_fs(training(cv));
    X_test = X_fs(test(cv), :);
    Y_test = Y_fs(test(cv));

    fun = @(trainX, trainY, testX, testY) mean(predict(fitcsvm(trainX, trainY, 'KernelFunction', 'RBF', 'Cost', costMatrix), testX) ~= testY);
    
    opts = statset('display', 'iter');
    [rankedFeatures, ~] = sequentialfs(fun, X_train, Y_train, 'cv', 5, 'options', opts);
    selectedFeatures_FS = find(rankedFeatures);
    
    svmModel = fitcsvm(X_train(:, selectedFeatures_FS), Y_train, 'KernelFunction', 'RBF');
    cvSVM = crossval(svmModel, 'KFold', 5);
    accuracy = 1 - kfoldLoss(cvSVM);
    
    predictions = predict(svmModel, X_test(:, selectedFeatures_FS));
    confMat = confusionmat(Y_test, predictions);
    TN = confMat(1,1); FP = confMat(1,2);
    FN = confMat(2,1); TP = confMat(2,2);
    
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
    precision = TP / (TP + FP);
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
end