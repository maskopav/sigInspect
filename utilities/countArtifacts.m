function countArtif = countArtifacts(X, Y, signalIds, verbose)
% Counts number of artifacts of types: POWER, BASE, FREQ.
%
% Inputs:
%   X          - Cell array of features per signal
%   Y          - Cell array of labels per signal (rows = artifact types)
%   signalIds  - Cell array of signal IDs (aligned with X and Y)
%   verbose    - (Optional) If true, prints the counts. Default: true
%
% Outputs:
%   countArtif (1-POW, 2-BASE, 3-FREQ)
    if nargin < 4
        verbose = true;
    end
    countArtif = containers.Map('KeyType', 'double', 'ValueType', 'double');

    % Extract binary artifact presence info per signal
    [~, artifact2, ~] = extractFeatureValues(X, Y, 2, signalIds); % POWER
    [~, artifact3, ~] = extractFeatureValues(X, Y, 3, signalIds); % BASE
    [~, artifact4, ~] = extractFeatureValues(X, Y, 4, signalIds); % FREQ

    % Convert to numeric and count
    countArtif(2)  = sum(double(string(artifact2)));
    countArtif(3) = sum(double(string(artifact3)));
    countArtif(4)  = sum(double(string(artifact4)));

    % Optional printout
    if verbose
        fprintf('POW artifacts:  %d\n', countArtif(2));
        fprintf('BASE artifacts: %d\n', countArtif(3));
        fprintf('FREQ artifacts: %d\n', countArtif(4));
    end
end
