function [net, predictedProbs] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, classWeights, lstmUnits, dropOut, maxEpochs, miniBatchSize, ...
    initialLearnRate, validationFrequency, validationPatience, classMode)
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
    %   classMode: 'binary' for binary classification, 'multi' for multi-label classification

    % Validate mode input
    if ~ismember(classMode, {'binary', 'multi'})
        error("Invalid mode. Use 'binary' for binary classification or 'multi' for multi-label classification.");
    end

        % Define common LSTM layers
    layers = [
        sequenceInputLayer(inputSize, "Name", "input")
        bilstmLayer(lstmUnits, "OutputMode", "sequence", "Name", "bilstm")
        dropoutLayer(dropOut, "Name", "dropout")
        fullyConnectedLayer(numClasses, "Name", "fc")
    ];
    
    % Configure output layers based on classMode
    if strcmp(classMode, 'binary')
        layers = [layers; 
            softmaxLayer("Name", "softmax");
            classificationLayer("Name", "output", "Classes", categorical(0:numClasses-1), "ClassWeights", classWeights) 
        ];
    else % Multi-label classification
        layers = [layers;
            sigmoidLayer("Name", "sigmoid");
            multilabelClassificationLayer("output", classWeights)
        ];
    end

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
    predictedProbs = predict(net, XTest, 'MiniBatchSize', miniBatchSize);
end
