function [accuracy, sensitivity, specificity, auc, optimal_threshold] = evaluateModel(predictedProbs, labels)
    % Ensure labels are numeric (0 or 1) for perfcurve
    labels = double(labels == "1"); 

    % Calculate ROC curve and AUC
    [fpr, tpr, thresholds, auc] = perfcurve(labels, predictedProbs, 1);

    % Find the index of the optimal threshold (e.g., Youden's J-statistic)
    [~, idx] = max(tpr - fpr); 
    optimal_threshold = thresholds(idx);

    % Determine predicted labels based on the optimal threshold
    predictedLabels = double(predictedProbs >= optimal_threshold);

    % Compute confusion matrix
    confusion_matrix = confusionmat(labels, predictedLabels);

    % Calculate performance metrics
    accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));
    sensitivity = confusion_matrix(2,2) / sum(confusion_matrix(2,:));
    specificity = confusion_matrix(1,1) / sum(confusion_matrix(1,:));

    % Plot confusion matrix
    figure;
    subplot(211)
    confusionchart(labels, predictedLabels,"RowSummary","row-normalized", ...
        "ColumnSummary","column-normalized", ...
        "Title",['Confussion matrix (accuracy = ', num2str(accuracy, '%.2f'),',threshold = ', num2str(optimal_threshold, '%.2f'), ')']);

    % Plot ROC curve
    subplot(212)
    plot(fpr, tpr)
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(['ROC Curve (AUC = ', num2str(auc, '%.2f'), ')']);
    hold on;
    plot([0 1], [0 1], '--');
    hold off;
    legend('ROC Curve', 'Random Guess');
    grid on;

end
