function YConverted = convertToBinaryLabels(Y, mode, maxN)
    % CONVERTTOBINARYLABELS - Converts numeric labels to either binary or multi-label format.
    %
    % Inputs:
    %   - Y: Cell array of numeric labels.
    %   - mode: String specifying conversion mode:
    %       * 'binary'  - Convert all nonzero labels to 1 (artifact vs. clean)
    %       * 'multi'   - Convert to multi-label binary format using annotNum2Bin
    %   - maxN: (Required for 'multi' mode) Maximum number of artifact types.
    %
    % Output:
    %   - YConverted: Cell array of converted labels (either binary or multi-label).
    
    % Validate mode input
    if ~ismember(mode, {'binary', 'multi'})
        error("Invalid mode. Use 'binary' for binary classification or 'multi' for multi-label classification.");
    end

    if strcmp(mode, 'multi') && nargin < 3
        error("For 'multi' mode, you must provide maxN (number of artifact types).");
    end

    % Apply conversion using cellfun
    if strcmp(mode, 'binary')
        YConverted = cellfun(@(x) categorical(double(double(string(x)) ~= 0)), Y, 'UniformOutput', false);
    else % mode == 'multi'
        YConverted = cellfun(@(x) annotNum2Bin(double(string(x)), maxN)', Y, 'UniformOutput', false);
    end
    
    fprintf("Labels converted using mode: %s\n", mode);
end
