function youdenIndex = computeYoudenIndex(trainX, trainY, testX, testY, costMatrix)
    model = fitcsvm(trainX, trainY, 'KernelFunction', 'RBF', 'Cost', costMatrix);
    predictions = predict(model, testX);

    % Ensure labels and predictions are of the same type
    if iscategorical(testY) && ~iscategorical(predictions)
        predictions = categorical(predictions);
    elseif isnumeric(testY) && ~isnumeric(predictions)
        testY = double(testY);
        predictions = double(predictions);
    end
 
    [~, sensitivity, specificity, ~, ~] = computeEvaluationMetrics(testY, predictions);
    youdenIndex = sensitivity + specificity - 1;
end


