function [evalMetrics, auc, optimalThreshold] = evaluateModel(predictedProbsCell, labelsCell, classMode)
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
    labels = cellfun(@(x) x', labelsCell, 'UniformOutput', false);
    labels = vertcat(labels{:});

    numClasses = size(labels, 2);      % Number of classes

    if strcmp(classMode, 'binary')
        % === Binary Classification ===
        positiveClassIdx = 2;  % Assuming second column corresponds to the positive class
        predictedProbs = cellfun(@(x) x(positiveClassIdx, :)', predictedProbsCell, 'UniformOutput', false);
        predictedProbs = vertcat(predictedProbs{:});
        
        labels = double(string(labels));

        % Compute ROC & AUC
        [fpr, tpr, thresholds, auc] = perfcurve(labels, predictedProbs, 1);

        % % Optimal threshold (Youden's J-statistic)
        % [~, idx] = max(tpr - fpr); 
        % optimalThreshold = thresholds(idx);

         % Compute Precision-Recall curve
        [prec, rec, thresholds, pr_auc] = perfcurve(labels, predictedProbs, 1, 'xCrit', 'reca', 'yCrit', 'prec');

        % Compute F1-score for each threshold
        f1_scores = 2 * (prec .* rec) ./ (prec + rec);
        f1_scores(isnan(f1_scores)) = 0; % Handle division by zero
        
        % Find the threshold with the highest F1-score
        [~, idx] = max(f1_scores);
        optimalThreshold = thresholds(idx);

        % Apply threshold
        predictedLabels = double(double(predictedProbs >= optimalThreshold));
        
        confusion_matrix = confusionmat(labels, predictedLabels, 'Order', [0 1]);
        evalMetrics = computeEvaluationMetrics(labels, predictedLabels);

        % Plot confusion matrix
        figure;
        subplot(2,2,1)
        confusionchart(confusion_matrix, ["Clean", "Artifact"], "RowSummary", "row-normalized", ...
            "ColumnSummary", "column-normalized", ...
            "Title", ['Confusion Matrix (Accuracy = ', num2str(evalMetrics.accuracy, '%.2f'), ...
                      ', F1 score = ', num2str(evalMetrics.f1, '%.2f'), ')']);

        % Plot ROC curve with optimal threshold
        subplot(2,2,2)
        plot(fpr, tpr, 'b', 'LineWidth', 1)
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title(['ROC Curve (AUC = ', num2str(auc, '%.2f'), ')']);
        hold on;
        plot(fpr(idx), tpr(idx), 'kx', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'LineWidth', 2);
        hold off;
        legend('ROC Curve', ['Optimal Threshold = ', num2str(optimalThreshold, '%.2f'), ],'Location', 'Best');
        grid on;

        % Plot probability distribution with threshold
        subplot(2,2,3);
        histogram(predictedProbs(labels == 0), 'BinWidth', 0.02, 'FaceColor', 'b', 'FaceAlpha', 0.5);
        hold on;
        histogram(predictedProbs(labels == 1), 'BinWidth', 0.02, 'FaceColor', 'r', 'FaceAlpha', 0.5);
        xline(optimalThreshold, '--k', 'LineWidth', 2);
        xlabel('Predicted Probability');
        ylabel('Frequency');
        title('Class Probability Distribution');
        legend('Class 0', 'Class 1', 'Threshold','Location', 'Best');
        grid on;

        % Plot Precision-Recall Curve
        subplot(2,2,4);
        plot(rec, prec, 'b', 'LineWidth', 1);
        xlabel('Recall');
        ylabel('Precision');
        title(['Precision-Recall Curve (AUC = ', num2str(pr_auc, '%.2f'), ')']);
        hold on;
        plot(rec(idx), prec(idx), 'kx', 'MarkerSize', 10, 'MarkerFaceColor', 'k', 'LineWidth', 2);
        hold off;
        grid on;
        

    elseif strcmp(classMode, 'multi')
        predictedProbs = cellfun(@(x) x', predictedProbsCell, 'UniformOutput', false);
        predictedProbs = vertcat(predictedProbs{:});

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
