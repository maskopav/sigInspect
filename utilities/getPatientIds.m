function [patientIds, numUniquePatients] = getPatientIds(signalIds)
    % Extracts patient IDs from signal IDs and returns the number of unique patients
    %
    % Inputs:
    % - signalIds: Cell array of signal IDs
    %
    % Outputs:
    % - patientIds: Cell array of patient IDs (everything before the first 't')
    % - numUniquePatients: Number of unique patients

    % Extract patient IDs (substring before the first 't')
    patientIds = cellfun(@(id) extractBefore(id, 't'), signalIds, 'UniformOutput', false);

    % Count unique patients
    numUniquePatients = numel(unique(patientIds));
end
