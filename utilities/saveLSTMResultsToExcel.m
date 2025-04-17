function saveLSTMResultsToExcel(artifactIdx, selectedFeatures_FS, evalMetricsTrain, evalMetricsVal, evalMetricsTest, lstmSettings, excelFile, sheetName)
    % Create results table with metrics from Train, Val, and Test sets
    resultsTable = table(artifactIdx, ...
        strjoin(string(selectedFeatures_FS), ', '), ...
        evalMetricsTrain.accuracy, evalMetricsTrain.sensitivity, evalMetricsTrain.specificity, evalMetricsTrain.f1, ...
        evalMetricsTrain.youden, evalMetricsTrain.prAUC, evalMetricsTrain.rocAUC, evalMetricsTrain.optimalThreshold, ...
        evalMetricsVal.accuracy, evalMetricsVal.sensitivity, evalMetricsVal.specificity, evalMetricsVal.f1, ...
        evalMetricsVal.youden, evalMetricsVal.prAUC, evalMetricsVal.rocAUC, evalMetricsVal.optimalThreshold, ...
        evalMetricsTest.accuracy, evalMetricsTest.sensitivity, evalMetricsTest.specificity, evalMetricsTest.f1, ...
        evalMetricsTest.youden, evalMetricsTest.prAUC, evalMetricsTest.rocAUC, evalMetricsTest.optimalThreshold, ...
        lstmSettings.lstmUnits, lstmSettings.dropOut, lstmSettings.maxEpochs, lstmSettings.miniBatchSize, ...
        lstmSettings.initialLearnRate, lstmSettings.classWeights(2), ... % Use binary class weight if applicable
        'VariableNames', {'artifactIdx', 'Selected_FS_Features', ...
        'Train_Accuracy', 'Train_Sensitivity', 'Train_Specificity', 'Train_F1', 'Train_Youden', 'Train_PR_AUC', 'Train_ROC_AUC', 'Train_Threshold', ...
        'Val_Accuracy', 'Val_Sensitivity', 'Val_Specificity', 'Val_F1', 'Val_Youden', 'Val_PR_AUC', 'Val_ROC_AUC', 'Val_Threshold', ...
        'Test_Accuracy', 'Test_Sensitivity', 'Test_Specificity', 'Test_F1', 'Test_Youden', 'Test_PR_AUC', 'Test_ROC_AUC', 'Test_Threshold', ...
        'LSTMUnits', 'Dropout', 'Epochs', 'BatchSize', 'LearnRate', 'ClassWeight'});

    % Save to Excel
    saveResultsToExcel(excelFile, sheetName, resultsTable);
end
