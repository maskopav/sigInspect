function [net, predictedProbs] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, classWeights, lstmUnits, dropOut, maxEpochs, miniBatchSize, ...
    initialLearnRate, validationFrequency, validationPatience)
    % This function trains an LSTM network and predicts probabilities.
    % 
    % Arguments:
    %   XTrain, YTrain, XVal, YVal, XTest, YTest: Training, validation, and test datasets
    %   inputSize: The number of features in the input data
    %   numClasses: The number of output classes (e.g., 2 for binary classification)
    %   classWeights: Vector of class weights for imbalanced data
    %   lstmUnits: The number of units in the LSTM layer
    %   dropOut: DropoutLayer
    %   maxEpochs: The maximum number of epochs to train the network
    %   miniBatchSize: The size of each mini-batch during training
    %   initialLearnRate: The initial learning rate for the optimizer
    %   validationFrequency: How often to run validation during training
    %   validationPatience: Number of epochs without improvement before stopping training

    % Define LSTM network architecture
    layers = [
        sequenceInputLayer(inputSize, "Name", "input")           % Input layer
        lstmLayer(lstmUnits, "OutputMode", "sequence", "Name", "lstm") % LSTM layer for sequence output
        dropoutLayer(dropOut)                                        % Dropout layer to prevent overfitting
        fullyConnectedLayer(numClasses, "Name", "fc")            % Fully connected layer
        softmaxLayer("Name", "softmax")                          % Softmax layer for probabilities
        classificationLayer("Name", "output", ...
            'Classes', categorical([0:numClasses-1]), ...         % Specify the classes explicitly
            'ClassWeights', classWeights)                        % Add class weights
    ];

    % Set training options
    options = trainingOptions("adam", ...
        "MaxEpochs", maxEpochs, ...
        "MiniBatchSize", miniBatchSize, ...
        'InitialLearnRate', initialLearnRate, ...
        "SequenceLength", "longest", ... % Automatically pad shorter sequences
        "Shuffle", "every-epoch", ...
        "Verbose", true, ...
        'GradientThreshold', 1, ...  
        "Plots", "training-progress", ...
        "ValidationData", {XVal, YVal}, ...
        "ValidationFrequency", validationFrequency, ...
        "ValidationPatience", validationPatience); % Stops training if validation loss doesn't improve after `validationPatience` epochs

    % Train the LSTM network
    net = trainNetwork(XTrain, YTrain, layers, options);

    % Predict probabilities on the test data
    probs = predict(net, XTest, 'MiniBatchSize', miniBatchSize);

    % Extract second row probabilities (for the positive class in binary classification)
    artifactProbs = cellfun(@(x) x(2,:)', probs, 'UniformOutput', false);
    predictedProbs = vertcat(artifactProbs{:});
end
