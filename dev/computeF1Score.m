function f1 = computeF1Score(trainX, trainY, testX, testY, costMatrix)
    model = fitcsvm(trainX, trainY, 'KernelFunction', 'RBF', 'Cost', costMatrix);
    predictions = predict(model, testX);

    evalMetrics = computeEvaluationMetrics(testY, predictions);
    f1 = evalMetrics.f1;
end
