function [selectedFeatures_FS, evalMetrics, svmModel] = featureSelectionWithUndersampling(X_fs, Y_fs, costMatrix, criteria, patientIds, cleanToArtifactRatio)
    % Default ratio if not provided
    if nargin < 6
        cleanToArtifactRatio = 2;
    end
    
    % Define evaluation function based on chosen criterion
    switch lower(criteria)
        case 'f1'
            baseEvalFunc = @(trainX, trainY, testX, testY) 1 - computeF1Score(trainX, trainY, testX, testY, costMatrix);
        case 'youden'
            baseEvalFunc = @(trainX, trainY, testX, testY) 1 - computeYoudenIndex(trainX, trainY, testX, testY, costMatrix);
        otherwise
            error('Invalid criterion. Choose from "f1" or "youden".');
    end
    
    % Create a container for the selected features
    numFeatures = size(X_fs, 2);
    selectedFeatures = false(1, numFeatures);

    % Number of folds
    nFolds = 5;
    
    % Set up 5-fold cross-validation based on patient IDs
    uniquePatients = unique(patientIds);
    numPatients = length(uniquePatients);
    fprintf('Number of patients: %d\n', numPatients);

    fprintf('Total number of signals: %d\n', size(X_fs, 1));
    fprintf('Original class distribution (0=clean, 1=artifact):\n');
    tabulate(Y_fs)

    cv = cvpartition(patientIds, 'KFold', 5); 
    
    % Forward feature selection
    bestScore = Inf;
    remainingFeatures = 1:numFeatures;
    
    disp('Starting forward feature selection with patient-level undersampling...');
    
    while ~isempty(remainingFeatures)
        currentBestScore = Inf;
        currentBestFeature = 0;
        
        % Try adding each remaining feature
        for i = 1:length(remainingFeatures)
            featureToTry = remainingFeatures(i);
            tempSelectedFeatures = selectedFeatures;
            tempSelectedFeatures(featureToTry) = true;
            
            % Evaluate using cross-validation with patient-level undersampling
            foldScores = zeros(nFolds, 1);
            
            for fold = 1:nFolds
                % Get patient indices for this fold
                trainIdx = training(cv, fold);
                testIdx = test(cv, fold);

                % Extract train/test data
                trainX = X_fs(trainIdx, tempSelectedFeatures);
                trainY = Y_fs(trainIdx);
                trainPatientIds = patientIds(trainIdx);
                testX = X_fs(testIdx, tempSelectedFeatures);
                testY = Y_fs(testIdx);
                % fprintf('Before undersampling, fold: %d\n', fold);
                % tabulate(trainY)

                % Perform undersampling on training data at patient level
                [balancedTrainX, balancedTrainY] = undersampleByRatio(trainX, trainY, trainPatientIds, cleanToArtifactRatio);
                % if fold == 1
                %     fprintf('After undersampling, fold: %d\n', fold);
                %     tabulate(balancedTrainY)
                % end

                % balancedTrainX = trainX;
                % balancedTrainY = trainY;

                % Evaluate on this fold
                foldScores(fold) = baseEvalFunc(balancedTrainX, balancedTrainY, testX, testY);
            end
            
            % Average score across folds
            avgScore = mean(foldScores);
            
            % Update best feature if this one is better
            if avgScore < currentBestScore
                currentBestScore = avgScore;
                currentBestFeature = featureToTry;
            end
        end
        
        % If no improvement, stop
        if currentBestScore >= bestScore
            break;
        end
        
        % Otherwise, add the best feature
        bestScore = currentBestScore;
        selectedFeatures(currentBestFeature) = true;
        remainingFeatures = setdiff(remainingFeatures, currentBestFeature);
        
        fprintf('Added feature %d, current score: %.4f, selected features: %d\n', ...
            currentBestFeature, bestScore, sum(selectedFeatures));
    end
    
    % Get indices of selected features
    selectedFeatures_FS = find(selectedFeatures);

    % Train SVM with probability estimates enabled
    svmModel = fitcsvm(X_fs(:, selectedFeatures_FS), Y_fs, ...
                      'Prior', 'uniform', ...
                      'KernelFunction', 'RBF', ...
                      'Standardize', true, ....
                      'Cost', costMatrix);

    [predictions, ~] = predict(svmModel, X_fs(:, selectedFeatures_FS));
    evalMetrics = computeEvaluationMetrics(Y_fs, predictions);

    % Convert to a model that can output probability scores
    svmProbModel = fitPosterior(svmModel);
    % To make predictions with soft labels (probabilities)
    [~, probScores] = predict(svmProbModel, X_fs(:, selectedFeatures_FS));
    % Compute PR AUC using the function
    prAUC = computePRCurveAUC(Y_fs, probScores(:,2), 1);
    evalMetrics.prAUC = prAUC;
    % Compute ROC AUC using the function
    rocAUC = computeROCAUC(Y_fs, probScores(:,2), 1);
    evalMetrics.rocAUC = rocAUC;
end
