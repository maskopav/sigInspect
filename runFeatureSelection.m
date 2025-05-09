clc; clear;

sigInspectAddpath;

% Parameters
% Define paths
dataFolder = 'data/';
csvFile = '_metadataMER2020.csv';
signalsFolder = 'signals/';

featureDataPath = fullfile(dataFolder, 'featureSetUndersampled.mat');

% Load features suitable for model training
% LOAD FEATURE DATA (X and Y)
fprintf('Feature file exist. Loading data...\n');
load(featureDataPath, 'featureSetUndersampled');

X = featureSetUndersampled.X;        % Cell array
Y = featureSetUndersampled.Y;        % Cell array
signalIds = featureSetUndersampled.signalIds; % Vector
featNames = featureSetUndersampled.featNames; % Cell array

% Data split for model training
ratios = struct('train', 0.65, 'val', 0.2, 'test', 0.15);
[trainIdx, valIdx, testIdx] = splitDataByPatients(signalIds, ratios);
[trainPatientIds, trainUniquePatients] = getPatientIds(signalIds(trainIdx));
[valPatientIds, valUniquePatients] = getPatientIds(signalIds(valIdx));
[testPatientIds, testUniquePatients] = getPatientIds(signalIds(testIdx));

% Display results
fprintf('Number of training samples: %d, number of unique patients: %d\n', numel(trainIdx), trainUniquePatients);
fprintf('Number of validation samples: %d, number of unique patients: %d\n', numel(valIdx), valUniquePatients);
fprintf('Number of test samples: %d, number of unique patients: %d\n', numel(testIdx), testUniquePatients);

% Final variables for the model
Xfinal = X;
Yfinal = Y;
%Yfinal = cellfun(@(y) categorical(double(y(:)')), Yfinal, 'UniformOutput', false);
signalIdsFinal = signalIds;

% Access the splits
XTrain = Xfinal(trainIdx, :);
YTrain = Yfinal(trainIdx, :);
XVal = Xfinal(valIdx, :);
YVal = Yfinal(valIdx, :);
XTest = Xfinal(testIdx, :);
YTest = Yfinal(testIdx, :);

%%
excelFile = 'FS_results_undersampling.xlsx';
sheetName = 'FS';
% Youden and recall similar results -> only Youden and F1 score
criteriaList = {'youden'};

%%% artifactIdx
%1   'clean'    'CLN'
%2    'power'    'POW'
%3    'baseline'    'BASE'
%4    'frequency artifact'    'FREQ'
%5    'irritated neuron'    'IRIT'
%6    'other'    'OTHR'
%7    'artifact'    'ARTIF'

for artifactIdx=2:4
    disp(artifactIdx)

    % Data preprocessing
    [X_fs_train, Y_fs_train, signal_ids_train] = extractFeatureValues(XTrain, YTrain, artifactIdx, trainPatientIds);

    % Subset for testing
    % X_fs_train = X_fs_train(1:1000,:);
    % Y_fs_train = Y_fs_train(1:1000);
    % signal_ids_train = signal_ids_train(1:1000);

    % Class weights - three different weights 
    YFsArtif = cellfun(@(y) y(artifactIdx, :), YTrain, 'UniformOutput',false);

    cleanToArtifactRatios = 1:0.25:5;
    for cleanToArtifactIdx= 1:length(cleanToArtifactRatios)
        cleanToArtifactRatio = cleanToArtifactRatios(cleanToArtifactIdx);
        alpha = 0.5;
        costWeight = computeClassWeights(Y_fs_train, alpha);
        costWeight = 1;
        costMatrix = [0 1; costWeight 0];
    
        for idx = 1:length(criteriaList)
            % Start time
            startTime = datetime('now');

            criteria = criteriaList{idx}; 
            disp(['Evaluating using criterion: ', criteria, ', Ratio: ', num2str(cleanToArtifactRatio)])

            % Feature selection with SVM RBF kernel
            [selectedFeatures_FS, evalMetrics_train, svmModel] = featureSelectionWithUndersampling(X_fs_train, Y_fs_train, costMatrix, criteria, signal_ids_train, cleanToArtifactRatio);
            %[selectedFeatures_FS, evalMetrics_train , svmModel] = featureSelection(X_fs_train, Y_fs_train, costMatrix, criteria, trainPatientIds);

            % Convert to a model that can output probability scores
            svmProbModel = fitPosterior(svmModel);

            % Predict on validation dataset, help to select cost function
            [X_fs_val, Y_fs_val, signal_ids_val] = extractFeatureValues(XVal, YVal, artifactIdx, valPatientIds);
            X_fs_val = X_fs_val(:, selectedFeatures_FS);

            predictions = predict(svmModel, X_fs_val);
            evalMetrics_val = computeEvaluationMetrics(Y_fs_val, predictions);

            % To make predictions with soft labels (probabilities)
            [~, probScores] = predict(svmProbModel, X_fs_val);
            % Compute PR AUC using the function
            prAUC = computePRCurveAUC(Y_fs_val, probScores(:,2), 1);
            evalMetrics_val.prAUC = prAUC;
            % Compute ROC AUC
            rocAUC = computeROCAUC(Y_fs_val, probScores(:,2), 1);
            evalMetrics_val.rocAUC = rocAUC;

            % Predict on unseen dataset, for feature comparison with LSTM
            [X_fs_unseen, Y_fs_unseen, signal_ids_test] = extractFeatureValues(XTest, YTest, artifactIdx, testPatientIds);
            X_fs_unseen = X_fs_unseen(:, selectedFeatures_FS);

            predictions = predict(svmModel, X_fs_unseen);
            evalMetrics_unseen = computeEvaluationMetrics(Y_fs_unseen, predictions);

            % To make predictions with soft labels (probabilities)
            [~, probScores] = predict(svmProbModel, X_fs_unseen);
            % Compute PR AUC using the function
            prAUC = computePRCurveAUC(Y_fs_unseen, probScores(:,2), 1);
            evalMetrics_unseen.prAUC = prAUC;
            % Compute ROC AUC
            rocAUC = computeROCAUC(Y_fs_unseen, probScores(:,2), 1);
            evalMetrics_unseen.rocAUC = rocAUC;


            % End timer
            endTime = datetime('now');
            duration = endTime - startTime;

            % Save results to Excel File
            resultsTable = table(artifactIdx, ...
                strjoin(string(selectedFeatures_FS), ', '), strjoin(string(featNames(selectedFeatures_FS)), ', '), ...
                evalMetrics_train.accuracy, evalMetrics_train.sensitivity, evalMetrics_train.specificity, evalMetrics_train.precision, evalMetrics_train.f1, evalMetrics_train.youden, evalMetrics_train.rocAUC, evalMetrics_train.prAUC, ...
                evalMetrics_val.accuracy, evalMetrics_val.sensitivity, evalMetrics_val.specificity, evalMetrics_val.precision, evalMetrics_val.f1, evalMetrics_val.youden, evalMetrics_val.rocAUC, evalMetrics_val.prAUC, ...
                evalMetrics_unseen.accuracy, evalMetrics_unseen.sensitivity, evalMetrics_unseen.specificity, evalMetrics_unseen.precision, evalMetrics_unseen.f1, evalMetrics_unseen.youden, evalMetrics_unseen.rocAUC, evalMetrics_unseen.prAUC, ...
                cleanToArtifactRatio, costWeight, string(criteria), ...
                startTime, duration, ...
                'VariableNames', {'artifactIdx', 'Selected_FS_Features', 'Selected_FS_Features_Names', ...
                                  'Accuracy_Train', 'Sensitivity_Train', 'Specificity_Train', 'Precision_Train', 'F1_Score_Train', 'Youden_Train', 'ROC_AUC_Train', 'PR_AUC_Train', ...
                                  'Accuracy_Validation', 'Sensitivity_Validation', 'Specificity_Validation', 'Precision_Validation', 'F1_Score_Validation', 'Youden_Validation', 'ROC_AUC_Validation', 'PR_AUC_Validation', ...
                                  'Accuracy_Unseen', 'Sensitivity_Unseen', 'Specificity_Unseen', 'Precision_Unseen', 'F1_Score_Unseen', 'Youden_Unseen', 'ROC_AUC_Unseen', 'PR_AUC_Unseen', ...
                                  'Clean_To_Artifact_Ratio', 'ClassWeight', 'Criterion', 'Start_Time', 'Duration'});
            saveResultsToExcel(excelFile, sheetName, resultsTable);
        end
    end
end