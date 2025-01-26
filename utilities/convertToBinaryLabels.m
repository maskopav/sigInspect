function YBinary = convertToBinaryLabels(Y)
    % CONVERT TO BINARY LABELS - Converts multiclass labels to binary.
    %
    % Input:
    %   - Y: Cell array of categorical labels (multiclass).
    %
    % Output:
    %   - YBinary: Cell array of categorical binary labels.
    
    YBinary = cell(size(Y));
    
    for i = 1:numel(Y)
        numericLabels = double(string(Y{i})); % Convert to numeric for manipulation
        numericLabels(numericLabels ~= 0) = 1; % Convert multiclass to binary
        YBinary{i, 1} = categorical(numericLabels); % Convert back to categorical
    end
end
