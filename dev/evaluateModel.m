function [accuracy, sensitivity, specificity, auc, optimalThreshold] = evaluateModel(predictedProbsCell, labelsCell, classMode)
    % Evaluate a classification model for binary and multi-label cases.
    % Handles cell inputs where each cell contains a matrix of values.
    %
    % Arguments:
    %   predictedProbsCell: Cell array where each cell contains a matrix of predicted probabilities.
    %   labelsCell: Cell array where each cell contains a binary matrix of true labels.
    %   classMode: 'binary' for binary classification, 'multi' for multi-label classification.
    %
    % Returns:
    %   accuracy: Average accuracy.
    %   sensitivity: Average sensitivity (TPR).
    %   specificity: Average specificity (TNR).
    %   auc: Macro-averaged AUC score.
    %   optimalThreshold: Best threshold per class.

    % Convert cell arrays to matrices
    predictedProbs = cellfun(@(x) x', predictedProbsCell, 'UniformOutput', false);
    predictedProbs = vertcat(predictedProbs{:});

    labels = cellfun(@(x) x', labelsCell, 'UniformOutput', false);
    labels = vertcat(labels{:});

    numClasses = size(labels, 2);      % Number of classes

    if strcmp(classMode, 'binary')
        % === Binary Classification ===
        positiveClassIdx = 2;  % Assuming second column corresponds to the positive class
        predictedProbs = predictedProbs(:, positiveClassIdx);
        labels = labels(:, positiveClassIdx);

        % Compute ROC & AUC
        [fpr, tpr, thresholds, auc] = perfcurve(labels, predictedProbs, 1);

        % Optimal threshold (Youden's J-statistic)
        [~, idx] = max(tpr - fpr); 
        optimalThreshold = thresholds(idx);

        % Apply threshold
        predictedLabels = double(predictedProbs >= optimalThreshold);

        % Compute confusion matrix
        confusion_matrix = confusionmat(labels, predictedLabels);

        % Compute performance metrics
        accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix(:));
        sensitivity = confusion_matrix(2,2) / sum(confusion_matrix(2,:));
        specificity = confusion_matrix(1,1) / sum(confusion_matrix(1,:));

        % Plot confusion matrix
        figure;
        subplot(2,1,1)
        confusionchart(confusion_matrix, ["Clean", "Artifact"], "RowSummary", "row-normalized", ...
            "ColumnSummary", "column-normalized", ...
            "Title", ['Confusion Matrix (Accuracy = ', num2str(accuracy, '%.2f'), ...
                      ', Threshold = ', num2str(optimalThreshold, '%.2f'), ')']);

        % Plot ROC curve
        subplot(2,1,2)
        plot(fpr, tpr)
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title(['ROC Curve (AUC = ', num2str(auc, '%.2f'), ')']);
        hold on;
        plot([0 1], [0 1], '--');
        hold off;
        legend('ROC Curve', 'Random Guess');
        grid on;

    elseif strcmp(classMode, 'multi')
        % === Multi-Label Classification ===
        auc = zeros(1, numClasses);
        optimalThreshold = zeros(1, numClasses);

        % Store per-class metrics
        sensitivity = zeros(1, numClasses);
        specificity = zeros(1, numClasses);
        accuracy = zeros(1, numClasses);

        figure;
        hold on;
        colors = lines(numClasses); % Generate distinct colors for classes

        for i = 1:numClasses
            % Compute ROC & AUC per class
            [fpr, tpr, thresholds, auc(i)] = perfcurve(labels(:, i), predictedProbs(:, i), 1);
            [~, idx] = max(tpr - fpr);
            optimalThreshold(i) = thresholds(idx);

            % Apply threshold per class
            predictedLabels = double(predictedProbs(:, i) >= optimalThreshold(i));

            % Compute confusion matrix
            tp = sum((predictedLabels == 1) & (labels(:, i) == 1));
            tn = sum((predictedLabels == 0) & (labels(:, i) == 0));
            fp = sum((predictedLabels == 1) & (labels(:, i) == 0));
            fn = sum((predictedLabels == 0) & (labels(:, i) == 1));

            % Compute per-class metrics
            accuracy(i) = (tp + tn) / (tp + tn + fp + fn);
            sensitivity(i) = tp / (tp + fn);  % Recall
            specificity(i) = tn / (tn + fp);  % True Negative Rate

            % Plot ROC per class
            plot(fpr, tpr, 'Color', colors(i, :), 'DisplayName', ['Class ', num2str(i)]);
        end

        % Compute macro-average metrics
        accuracy = mean(accuracy);
        sensitivity = mean(sensitivity);
        specificity = mean(specificity);
        auc = mean(auc);  % Macro-averaged AUC

        % Finalize ROC plot
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title(['Multi-Label ROC Curves (Macro-AUC = ', num2str(auc, '%.2f'), ')']);
        legend show;
        plot([0 1], [0 1], '--k');  % Random baseline
        grid on;
        hold off;

        figure;
        for i = 1:numClasses
            predictedLabels = double(predictedProbs(:, i) >= optimalThreshold(i));
            cm = confusionmat(labels(:, i), predictedLabels);
            subplot(ceil(sqrt(numClasses)), ceil(sqrt(numClasses)), i); % Arrange in a grid
            confusionchart(cm, ["Negative", "Positive"], 'Title', ['Class ' num2str(i)]);
        end

    else
        error("Invalid classMode. Use 'binary' or 'multi'.");
    end
end
