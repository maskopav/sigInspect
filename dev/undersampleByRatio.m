function [X_balanced, Y_balanced] = undersampleByRatio(X, Y, patientIds, cleanToArtifactRatio, method)
    % Function to undersample clean samples to balance with artifact samples
    % Inputs:
    %   X - Features matrix
    %   Y - Labels (categorical or numeric, with 0=clean and 1=artifact)
    %   patientIds - Cell array or array of patient identifiers
    %   cleanToArtifactRatio - Desired ratio of clean to artifact samples
    %   method - 'random' for global random undersampling or 'patient' for patient-specific
    
    % If method is not specified, default to 'random'
    if nargin < 5
        method = 'random';
    end
    
    % Convert Y to categorical if it's not already
    if ~iscategorical(Y)
        Y = categorical(Y);
    end
    
    % Get unique classes
    classes = categories(Y);
    if length(classes) ~= 2
        error('This function expects exactly 2 classes (clean and artifact)');
    end
    
    % Determine which class is clean and which is artifact
    cleanLabel = categorical(0);  % Assuming 0 = clean, 1 = artifact
    artifactLabel = categorical(1);
    
    % Get counts for each class
    summaryY = countcats(Y);
    
    % Calculate how many clean samples to keep
    numCleanSamples = sum(Y == cleanLabel);
    numArtifactSamples = sum(Y == artifactLabel);
    numCleanToKeep = min(numCleanSamples, round(cleanToArtifactRatio * numArtifactSamples));
    numDeleteClean = numCleanSamples - numCleanToKeep;
    
    % If no samples need to be deleted, return original data
    if numDeleteClean <= 0
        X_balanced = X;
        Y_balanced = Y;
        return;
    end
    
    % Perform undersampling based on the selected method
    if strcmpi(method, 'random')
        % METHOD 1: Random undersampling across all samples
        % Find indices of clean samples
        cleanIndices = find(Y == cleanLabel);
        
        % Randomly select clean samples to delete
        deleteIndices = cleanIndices(randperm(length(cleanIndices), numDeleteClean));
        
        % Create a mask for samples to keep
        keepMask = true(size(Y));
        keepMask(deleteIndices) = false;
        
        % Create balanced dataset
        X_balanced = X(keepMask, :);
        Y_balanced = Y(keepMask);
        
    elseif strcmpi(method, 'patient')
        % METHOD 2: Patient-specific undersampling
        
        % Get unique patients
        uniquePatients = unique(patientIds);
        
        % Initialize mask for samples to keep
        keepMask = true(size(Y));
        
        % Process each patient separately
        for i = 1:length(uniquePatients)
            patientMask = strcmp(patientIds, uniquePatients{i});
            patientY = Y(patientMask);
            
            % Count clean and artifact samples for this patient
            patientCleanCount = sum(patientY == cleanLabel);
            patientArtifactCount = sum(patientY == artifactLabel);
            
            % Skip if no clean samples or no artifact samples
            if patientCleanCount == 0 || patientArtifactCount == 0
                continue;
            end
            
            % Calculate how many clean samples to keep for this patient
            patientKeepClean = min(patientCleanCount, round(cleanToArtifactRatio * patientArtifactCount));
            patientDeleteClean = patientCleanCount - patientKeepClean;
            
            % If deletion needed for this patient
            if patientDeleteClean > 0
                % Find indices of clean samples for this patient
                patientIndices = find(patientMask);
                patientCleanIndices = patientIndices(patientY(patientIndices - sum(patientMask) + 1) == cleanLabel);
                
                % Randomly select clean samples to delete for this patient
                patientDeleteIndices = patientCleanIndices(randperm(length(patientCleanIndices), patientDeleteClean));
                
                % Update the keep mask
                keepMask(patientDeleteIndices) = false;
            end
        end
        
        % Create balanced dataset
        X_balanced = X(keepMask, :);
        Y_balanced = Y(keepMask);
        
    else
        error('Unknown method. Use ''random'' or ''patient''');
    end
end