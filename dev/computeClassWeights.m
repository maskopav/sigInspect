function classWeights = computeClassWeights(Yfinal, alpha)
    % Computes inverse frequency class weights for multi-label classification.
    %
    % Arguments:
    %   Yfinal - Cell array of binary label matrices (numClasses x numSamples)
    %   alpha - Exponent factor (default = 1.5, higher means more weight on rare classes)
    %
    % Returns:
    %   classWeights - Vector of weights for each class

    % Get number of classes
    numClasses = size(Yfinal{1}, 1); % Number of output classes
    numSamples = length(Yfinal); % Total number of samples

    % Initialize vector to store class occurrence counts
    classCounts = zeros(numClasses, 1);

    % Count occurrences of each class 
    for i = 1:numSamples
        classCounts = classCounts + sum(Yfinal{i}, 2); 
    end

    % Compute exponential inverse frequency class weights
    classWeights = (numSamples ./ (classCounts + eps)).^alpha;

    % Normalize weights to sum to numClasses
    classWeights = classWeights / sum(classWeights) * numClasses;

    % Display the computed class weights
    disp("Computed class weights:");
    disp(classWeights);
end
