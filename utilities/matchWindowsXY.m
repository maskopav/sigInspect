function Y_matched = matchWindowsXY(X, Y)
    % Matches the number of windows in Y to the number of windows in X.
    %
    % Inputs:
    %   - X: Cell array where each cell is a matrix [features x windows].
    %   - Y: Cell array where each cell is a logical [artifacts x windows].
    %
    % Outputs:
    %   - Y_matched: Cell array Y with the number of windows matched to X.

    % Check if the number of signals (cells) in X and Y is the same
    if numel(X) ~= numel(Y)
        error('X and Y must have the same number of cells.');
    end

    Y_matched = cell(size(Y)); % Preallocate the output cell array

    % Iterate over each signal (cell)
    for i = 1:numel(X)
        num_windows_X = size(X{i}, 2); % Get the number of windows in X{i}

        if size(Y{i}, 2) > num_windows_X
            % If Y has more windows than X, truncate Y
            Y_matched{i} = Y{i}(:, 1:num_windows_X);
        elseif size(Y{i}, 2) < num_windows_X
            % If Y has less windows than X, pad Y with some value
            %  Y_matched{i} = [Y{i}, zeros(size(Y{i}, 1), num_windows_X - size(Y{i}, 2))]; % Pad with zeros
             error('Y has less windows than X, this case is not handled.');
        else
            % If Y has the same number of windows as X, no change needed
            Y_matched{i} = Y{i};
        end
    end
end
