function [net, predictedProbsTrain, predictedProbsVal, predictedProbsTest] = trainAndPredictLSTM(XTrain, YTrain, XVal, YVal, XTest, YTest, ...
    inputSize, numClasses, lstmSettings, classMode)
    % This function trains an LSTM network and predicts probabilities.

    % Validate mode input
    if ~ismember(classMode, {'binary', 'multi'})
        error("Invalid mode. Use 'binary' for binary classification or 'multi' for multi-label classification.");
    end

    % Extract settings from struct
    lstmUnits = lstmSettings.lstmUnits;
    dropOut = lstmSettings.dropOut;
    maxEpochs = lstmSettings.maxEpochs;
    miniBatchSize = lstmSettings.miniBatchSize;
    initialLearnRate = lstmSettings.initialLearnRate;
    classWeights = lstmSettings.classWeights;
    validationFrequency = lstmSettings.validationFrequency;
    validationPatience = lstmSettings.validationPatience;

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

    % Predict probabilities
    predictedProbsTrain = predict(net, XTrain, 'MiniBatchSize', miniBatchSize);
    predictedProbsVal   = predict(net, XVal, 'MiniBatchSize', miniBatchSize);
    predictedProbsTest  = predict(net, XTest, 'MiniBatchSize', miniBatchSize);
end
