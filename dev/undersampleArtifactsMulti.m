function [X_new, Y_new, signalIds_new] = undersampleArtifactsMulti(X, Y, signalIds, config)
% Multi-artifact undersampling with two-stage removal and cross-type iteration.
% Config: struct array with fields - artifactIdx, nToRemove, patientId
% REQUIREMENTS:
% Undersample all artifact types in one pass using a config struct with artifactIdx, nToRemove, patientId (array of structs).
% Two-stage undersampling:
% Stage 1: Randomly remove whole signals containing many artifacts (40% of signal) of the target type (50%) to avoid bias.
% Stage 2: Trim artifacts within signals from start or end to meet the target.
% Prefer windows with only the current artifact (if not from shared patient).
% Avoid trimming artifacts shared across types for the same patient.
% Randomly alternates between artifact types during removal.

    % Initialize everything
    rng(1)
    X_new = X;
    Y_new = Y;
    signalIds_new = signalIds;
    alreadyTrimmed = zeros(length(signalIds_new), 1);
    [patientId, ~] = getPatientIds(signalIds_new);
    wholeSignalRemoveRatio = 0.7;
    
    disp('Artifacts counts at the beginning:')
    % Assuming countArtifacts returns artifact counts in the order they appear in Y matrix rows
    [artifactCounts] = countArtifacts(X_new, Y_new, signalIds_new);
    
    % Initialize per-artifact counters
    remainingToRemove = containers.Map('KeyType', 'double', 'ValueType', 'double');
    fullTargetCount = containers.Map('KeyType', 'double', 'ValueType', 'double');  % How many removed by full signals
    patientsPerArt = containers.Map('KeyType', 'double', 'ValueType', 'any');
    totalRemovedMap = containers.Map('KeyType', 'double', 'ValueType', 'double');  % For logging
    
    artifactTypes = [];  % Collect all artifact indices from config
    
    % Process configuration and set up the tracking
    for c = 1:numel(config)
        aid = config(c).artifactIdx;  % These are 2, 3, or 4
        artifactTypes(end+1) = aid;
        
        % Initialize counts
        if isKey(artifactCounts, aid)
            fprintf('Artifact %d: %d instances\n', aid, artifactCounts(aid));
        else
            fprintf('Warning: Artifact %d not found in data\n', aid);
            artifactCounts(aid) = 0;
        end
        
        remainingToRemove(aid) = config(c).nToRemove;
        fullTargetCount(aid) = 0;  % Counter for signals fully removed
        patientsPerArt(aid) = config(c).patientId;
        totalRemovedMap(aid) = 0;  % Total artifacts removed of this type
        
        fprintf('Target to remove for artifact %d: %d\n', aid, config(c).nToRemove);
    end
    
    fprintf('\n=== Stage 1: Randomly remove full signals (≥40%% artifact, 50%% chance) ===\n');
    
    % Stage 1 loop - continue until we've removed 50% of our target removals with full signals
    iterCount = 0;
    maxIter = 10000;  % Safety limit
    
    while iterCount < maxIter 
        iterCount = iterCount + 1;
        
        % Choose artifact type that still needs removal
        eligibleAids = [];
        for aidx = 1:length(artifactTypes)
            aid = artifactTypes(aidx);
            % Check if we still need to remove more artifacts of this type via full signals
            if fullTargetCount(aid) < ceil(remainingToRemove(aid) * wholeSignalRemoveRatio)
                eligibleAids(end+1) = aid;
            end
        end
        
        if isempty(eligibleAids)
            fprintf('All artifact types have met their stage 1 targets.\n');
            break;  % No more eligible artifact types, stage 1 complete
        end
        
        % Randomly select an artifact type to process
        aid = eligibleAids(randi(length(eligibleAids)));
        
        % Find candidates not yet trimmed
        candidates = find(~alreadyTrimmed);
        
        if isempty(candidates)
            fprintf('No more untrimmed signals available.\n');
            break;  % No more signals available
        end
        
        % Select random candidate
        k = candidates(randi([1 length(candidates)],1));
        if isempty(Y_new{k})
            continue;
        end
        
        y_mat = Y_new{k};  % matrix of shape [nArtifacts x nWindows]
        if size(y_mat, 1) < aid
            continue;  % Skip if this signal doesn't have enough rows
        end
        
        y_art = y_mat(aid, :);  % Get the artifact data from the specific row
        if all(y_art == 0)
            continue;
        end
        
        winCount = size(y_mat, 2);
        n_artifact = sum(y_art);  % Count windows with this artifact
        
        % If this signal has enough artifacts of this type and passes random check
        if n_artifact >= 0.3 * winCount && rand < wholeSignalRemoveRatio
            pid = patientId{k};
            % Check if signal is from a patient we want to undersample for this artifact
            targetPatients = patientsPerArt(aid);
            if any(strcmp(pid, targetPatients))
                % Remove entire signal
                fprintf('Removing signal %d with %d artifacts of type %d (patient %s)\n', ...
                       k, n_artifact, aid, pid);
                % Track the removal for ALL artifact types in this signal
                removeSignal = true;
                [fullTargetIndicies] = setdiff(artifactTypes, eligibleAids);
                if ~isempty(fullTargetIndicies)
                    if sum(y_mat(fullTargetIndicies,:)) > 0
                        removeSignal = false;
                    end
                end
                
                if removeSignal
                    for a = artifactTypes
                        % Only process if this artifact type exists in the signal
                        if size(y_mat, 1) >= a
                            removedCount = sum(y_mat(a, :));
                            
                            % Update counters for this artifact type
                            if isKey(fullTargetCount, a)
                                fullTargetCount(a) = fullTargetCount(a) + removedCount;
                            end
                            
                            if isKey(totalRemovedMap, a)
                                totalRemovedMap(a) = totalRemovedMap(a) + removedCount;
                            end
                            
                            fprintf('  Removed %d instances of artifact type %d\n', removedCount, a);
                        end
                    end
                
                    X_new{k} = [];
                    Y_new{k} = [];
                    signalIds_new{k} = [];
                    alreadyTrimmed(k) = true;
                end
                
                % Check if some of artifact types has met stage 1 targets
                for a = artifactTypes
                    if fullTargetCount(a) >= ceil(remainingToRemove(a) * wholeSignalRemoveRatio)
                        fprintf('Artifact type %d have met their stage 1 targets after this removal.\n', a);
                    end
                end
            end
        end
    end
    
    if iterCount >= maxIter
        fprintf('Warning: Maximum iteration count reached. Stage 1 may not be complete.\n');
    end
    
    % Clean up empty cells
    emptyIdx = cellfun(@isempty, X_new);
    X_new = X_new(~emptyIdx);
    Y_new = Y_new(~emptyIdx);
    signalIds_new = signalIds_new(~emptyIdx);
    
    % Update artifact counts after stage 1
    [artifactCountsAfter] = countArtifacts(X_new, Y_new, signalIds_new, false);
    
    % Display stage 1 results
    fprintf('\nStage 1 completed:\n');
    for aid = artifactTypes
        origCount = artifactCounts(aid);
        newCount = 0;
        if isKey(artifactCountsAfter, aid)
            newCount = artifactCountsAfter(aid);
        end
        removedCount = origCount - newCount;
        fprintf('Artifact %d: %d -> %d (removed %d)\n', aid, origCount, newCount, removedCount);
    end
    
    % Update remaining targets for stage 2
    for aid = artifactTypes
        if isKey(totalRemovedMap, aid)
            remainingToRemove(aid) = max(0, config(find([config.artifactIdx] == aid)).nToRemove - totalRemovedMap(aid));
            fprintf('Remaining to remove for artifact %d: %d\n', aid, remainingToRemove(aid));
        end
    end 

    fprintf('\n=== Stage 2: Trim artifacts from signals (start/end) ===\n');

    maxIterStage2 = 5000;
    stage2Iter = 0;
    [patientId, ~] = getPatientIds(signalIds_new);
    nonTrimmableSignals = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for aid = artifactTypes
        nonTrimmableSignals(aid) = [];
    end

    while stage2Iter < maxIterStage2
        stage2Iter = stage2Iter + 1;

        % Recalculate remaining to remove after each iteration
        [artifactCounts_new] = countArtifacts(X_new, Y_new, signalIds_new, false);
        for aid = artifactTypes
            if isKey(artifactCounts, aid)
                remainingToRemove(aid) = max(0, config(find([config.artifactIdx] == aid)).nToRemove - artifactCounts(aid) + artifactCounts_new(aid));
                % fprintf('Remaining to remove for artifact %d: %d\n', aid, remainingToRemove(aid));
            end
        end 

        % Display removal status every 500 iterations
        if mod(stage2Iter, 200) == 0
            for aid = artifactTypes
                fprintf('Iter %d, Remaining to remove for artifact %d: %d\n', stage2Iter, aid, remainingToRemove(aid));
            end
        end

        % Stop if all are done
        if all(arrayfun(@(a) remainingToRemove(a) <= 0, artifactTypes))
            fprintf('All artifact types have reached target after trimming.\n');
            break;
        end

        % Pick a random artifact type that still needs trimming
        eligibleAids = artifactTypes(arrayfun(@(a) remainingToRemove(a) > 0, artifactTypes));
        if isempty(eligibleAids)
            break;
        end
        aid = eligibleAids(randi(length(eligibleAids)));

        % Get list of signals belonging to the correct patients
        allowedPatients = config(find([config.artifactIdx] == aid)).patientId;
        allowedSignalIdx = find(ismember(patientId, allowedPatients));

        if isempty(allowedSignalIdx)
            disp('No signal ids found for selected patient.')
            continue;
        end

        % Random signal and channel
        k = allowedSignalIdx(randi(length(allowedSignalIdx)));
        if ismember(k, nonTrimmableSignals(aid))
            continue;
        end

        y_mat = Y_new{k};
        if size(y_mat, 1) < aid
            continue;
        end
        y_art = y_mat(aid, :);

        % Skip if no artifact of this type
        if all(y_art == 0)
            continue;
        end

        % Decide trim direction randomly
        int = rand(1,1);
        if int < 0.5
            trimDir = "start";
            trimIdx = find(y_art == 1, 1, 'first');
            trimRange = 1:trimIdx;
        else
            trimDir = "end";
            trimIdx = find(y_art == 1, 1, 'last');
            trimRange = trimIdx:length(y_art);
        end

        % Do not trim if signal would go below 5 windows
        n_remaining = length(y_art) - length(trimRange);
        if n_remaining < 5
            if int >= 0.5
                trimDir = "start";
                trimIdx = find(y_art == 1, 1, 'first');
                trimRange = 1:trimIdx;
            else
                trimDir = "end";
                trimIdx = find(y_art == 1, 1, 'last');
                trimRange = trimIdx:length(y_art);
            end

            % Do not trim if signal would go below 5 windows
            n_remaining = length(y_art) - length(trimRange);
            if n_remaining < 5
                nonTrimmableSignals(aid) = unique([nonTrimmableSignals(aid), k]);
                fprintf('Non trimmable signals for %d artifact:\n', aid)
                disp(nonTrimmableSignals(aid));
                continue;
            end
        end

        % Check if other already-fulfilled artifacts exist at trimIdx
        fulfilledAids = artifactTypes(arrayfun(@(a) remainingToRemove(a) <= 0 && a ~= aid, artifactTypes));
        skip = false;
        for otherAid = fulfilledAids
            if size(y_mat, 1) >= otherAid && sum(y_mat(otherAid, trimRange)) > 0
                skip = true;
                break;
            end
        end
        if skip
            continue;
        end

        % Trim
        fprintf('Trimming %s of signal %d (removing %d windows)\n', trimDir, k, length(trimRange));
        X_new{k}(:, trimRange) = [];
        Y_new{k}(:, trimRange) = [];
        signalIds_new{k}(trimRange) = [];
        for aid = artifactTypes
            removedNow = sum(y_mat(aid, trimRange));
            totalRemovedMap(aid) = totalRemovedMap(aid) + removedNow;
        end

        % Clean up empty cells
        emptyIdx = cellfun(@isempty, X_new);
        X_new = X_new(~emptyIdx);
        Y_new = Y_new(~emptyIdx);
        signalIds_new = signalIds_new(~emptyIdx);
    end

    if stage2Iter >= maxIterStage2
        fprintf('Warning: Max iterations reached in Stage 2.\n');
    end

    % Final report
    fprintf('\nFinal artifact counts after trimming:\n');
    finalCounts = countArtifacts(X_new, Y_new, signalIds_new, false);
    for aid = artifactTypes
        orig = artifactCounts(aid);
        final = 0;
        if isKey(finalCounts, aid)
            final = finalCounts(aid);
        end
        fprintf('Artifact %d: %d -> %d (removed %d)\n', aid, orig, final, orig - final);
    end
end
