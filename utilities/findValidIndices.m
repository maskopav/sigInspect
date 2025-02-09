function validIndices = findValidIndices(data)
    % FINDVALIDINDICES - Returns indices of non-NaN sequences in a cell array.
    %
    % Inputs:
    %   - data: Cell array of sequences (either feature variables or raw signal data).
    %
    % Outputs:
    %   - validIndices: Logical array where true means the entry should be kept.

    % Identify valid entries (without NaNs)
    validIndices = cellfun(@(x) ~any(isnan(x(:))), data);
    
    fprintf("Removing %d sequences with NaN values. Remaining sequences: %d\n", ...
        numel(data) - sum(validIndices), sum(validIndices));
end
