function [trainIdx, valIdx, testIdx] = splitDataByPatients(signalIds, ratios)
    % Splits data into train, validation, and test sets by patient IDs
    %
    % Inputs:
    % - signalIds: Cell array of signal IDs
    % - ratios: Struct with fields 'train', 'val', and 'test' specifying split ratios
    %
    % Outputs:
    % - trainIdx: Indices of training set
    % - valIdx: Indices of validation set
    % - testIdx: Indices of test set

    % Extract patient IDs and unique patient list
    [patientIds, ~] = getPatientIds(signalIds);
    uniquePatientIds = unique(patientIds);

    % Shuffle unique patient IDs for random splitting
    rng(20); % For reproducibility
    shuffledPatientIds = uniquePatientIds(randperm(length(uniquePatientIds)));

    % Compute number of patients for each split
    numPatients = length(shuffledPatientIds);
    numTrain = round(ratios.train * numPatients);
    numVal = round(ratios.val * numPatients);
    numTest = numPatients - numTrain - numVal;

    % Assign patients to train, validation, and test groups
    trainPatientIds = shuffledPatientIds(1:numTrain);
    valPatientIds = shuffledPatientIds(numTrain + 1:numTrain + numVal);
    testPatientIds = shuffledPatientIds(numTrain + numVal + 1:end);

    % Get indices for each group
    trainIdx = find(ismember(patientIds, trainPatientIds));
    valIdx = find(ismember(patientIds, valPatientIds));
    testIdx = find(ismember(patientIds, testPatientIds));
end
