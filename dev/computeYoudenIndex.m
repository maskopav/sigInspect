function youdenIndex = computeYoudenIndex(trainX, trainY, testX, testY, costMatrix)
    model = fitcsvm(trainX, trainY, 'Prior', 'uniform', 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto'); %, 'Cost', costMatrix);
    predictions = predict(model, testX);

    % Ensure labels and predictions are of the same type
    if iscategorical(testY) && ~iscategorical(predictions)
        predictions = categorical(predictions);
    elseif isnumeric(testY) && ~isnumeric(predictions)
        testY = double(testY);
        predictions = double(predictions);
    end
 
    evalMetrics = computeEvaluationMetrics(testY, predictions);
    sensitivity = evalMetrics.sensitivity;
    specificity = evalMetrics.specificity;
    youdenIndex = sensitivity + specificity - 1;
end


