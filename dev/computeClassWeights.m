function classWeights = computeClassWeights(Yfinal, alpha)
    % Computes inverse frequency class weights for multi-label classification.
    %
    % Arguments:
    %   Yfinal - Cell array of binary label matrices (numClasses x numSamples)
    %   alpha - Exponent factor (default = 1.5, higher means more weight on rare classes)
    %
    % Returns:
    %   classWeights - Vector of weights for each class

    if nargin < 2
        alpha = 1.5; % Default alpha if not provided
    end

    % Convert categorical to numeric if needed
    Yfinal = cellfun(@(y) double(string(y)), Yfinal, 'UniformOutput', false);

    numClasses = size(Yfinal{1}, 1); % Number of output classes
    numSamples = sum(cellfun(@(y) size(y, 2), Yfinal)); % Total number of samples
    % Initialize vector to store class occurrence counts
    classCounts = zeros(numClasses, 1);

    % Count occurrences of each class 
    for i = 1:length(Yfinal)
        classCounts = classCounts + sum(Yfinal{i}, 2); % Sum over all samples 
    end

    % Compute exponential inverse frequency class weights
    classWeights = (numSamples ./ (classCounts + eps)).^alpha;

    % Normalize weights to sum to numClasses
    if numClasses > 1
        classWeights = classWeights / sum(classWeights) * numClasses;
    end

    % Display the computed class weights
    disp("Computed class weights:");
    disp(classWeights);
end
