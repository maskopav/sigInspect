function counts = countArtifactPerPatient(artifactVec, signalIds)
    % Count number of artifacts per unique patient and return struct
    uniqueIDs = cellstr(unique(signalIds));  % Fix: ensure it's a cell array
    artifactCounts = zeros(size(uniqueIDs));
    
    for i = 1:numel(uniqueIDs)
        % Select signals belonging to this patient
        patientSignals = strcmp(signalIds, uniqueIDs{i});
        % Sum up artifacts for those signals
        artifactCounts(i) = sum(double(string(artifactVec(patientSignals))));
    end

    counts.patientIds = uniqueIDs;
    counts.artifactCounts = artifactCounts;

end