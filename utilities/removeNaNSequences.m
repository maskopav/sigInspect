function [XClean, YClean, signalIdsClean] = removeNaNSequences(X, Y, signalIds)
    % REMOVE NAN SEQUENCES - Removes sequences from X and Y containing NaN values.
    %
    % Inputs:
    %   - X: Cell array of feature sequences.
    %   - Y: Cell array of corresponding labels.
    %   - signalIds: Cell array of corresponding signalIds.
    %
    % Outputs:
    %   - XClean: Cell array of feature sequences without NaN values.
    %   - YClean: Cell array of labels corresponding to cleaned features.
    %   - signalIdsClean: Cell array of signalIds.

    XClean = {};
    YClean = {};
    signalIdsClean = {};
    
    for i = 1:numel(X)
        if ~any(isnan(X{i}), 'all')
            XClean{end+1, 1} = X{i};
            YClean{end+1, 1} = Y{i};
            signalIdsClean{end+1, 1} = signalIds{i};
        else
            fprintf("Skipping sequence %d: Contains NaN values.\n", i);
        end
    end
    
    fprintf("Removing %d sequences with NaN values. Remaining sequences: %d\n", ...
        numel(X) - numel(XClean), numel(XClean));
end
