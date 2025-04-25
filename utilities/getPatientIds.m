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
    % patientIds = cellfun(@(id) extractBefore(id, 't'), signalIds, 'UniformOutput', false);
    patientIds = cell(size(signalIds)); 

    for i = 1:numel(signalIds)
        signalId = signalIds{i};
        chxPattern = regexp(signalId, 'ch\d+$', 'once'); % Find 'ch' followed by digits at the end

        if ~isempty(chxPattern)
            % Case 1: 'chx' pattern found (e.g., 'sig_Kos_JurM_Dex_1_1_ch1')
            % Extract characters from 5th to 12th (inclusive)
            if length(signalId) >= 12
                patientIds{i} = signalId(5:12);
            else
                 patientIds{i} = ''; % Or some other default value
                 warning('Signal ID "%s" is too short to extract patient ID.', signalId);
            end
        else
            % Case 2: 'chx' pattern not found (e.g., '59t146p4x')
            patientIds{i} = extractBefore(signalId, 't');
        end
    end

    % Count unique patients
    numUniquePatients = numel(unique(patientIds));
end
